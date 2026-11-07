# fetch_model.cr — download a static embedding model and extract it.
#
#   crystal run --release tools/fetch_model.cr multilingual
#   crystal run --release tools/fetch_model.cr base --out models
#   crystal run --release tools/fetch_model.cr --list
#
# The models themselves are not redistributed with semrember: they are large
# and they belong to their authors.  This tool fetches them from the
# Hugging Face hub and converts them into semrember's own M2VE/M2VS binary
# format, so the runtime needs nothing — no model2vec, no tokenizers, no
# numpy.
#
# It is idempotent: if the target file is already present and its SHA-256
# matches, nothing is downloaded.
#
# The extraction is verified, not assumed: the output's SHA-256 is checked
# against the published checksum, and a mismatch is a hard failure rather
# than a silently installed file.

require "http/client"
require "uri"
require "json"
require "digest/sha256"
require "file_utils"
require "../src/semrember"

module FetchModel
  extend self

  HF        = "https://huggingface.co"
  HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"

  record Spec,
    name : String,
    repo : String,
    file : String,
    sha256 : String,
    kind : String

  # The published checksums are of the *extracted* .bin files, so the whole
  # download-and-convert pipeline is covered by them.
  SPECS = {
    "base" => Spec.new(
      "base",
      "minishlab/potion-base-8M",
      "potion-base-8M.bin",
      "8e920b34f4c3aab17242de3418d764d808dfb23eda732775e7a898231937e70c",
      "M2VE"
    ),
    "multilingual" => Spec.new(
      "multilingual",
      "minishlab/potion-multilingual-128M",
      "potion-multilingual-128M.bin",
      "8df9483603872fee5a1f628c595b7f4e6b5b38400f40a4c694ebb51385ae657a",
      "M2VS"
    ),
  }

  # ── downloading ────────────────────────────────────────────────────────
  def resolve(repo : String, file : String) : String
    "https://huggingface.co/#{repo}/resolve/main/#{file}"
  end

  # The Hugging Face resolve endpoint answers with a redirect to a CDN, and
  # Crystal's HTTP client does not follow redirects on its own, so this does.
  def download(url : String, dest : String, label : String)
    FileUtils.mkdir_p(File.dirname(dest))
    part = dest + ".part"
    current = url
    redirects = 0
    loop do
      target = nil.as(String?)
      total = 0_i64
      HTTP::Client.get(current) do |response|
        code = response.status_code
        if code == 301 || code == 302 || code == 303 || code == 307 || code == 308
          target = response.headers["Location"]?
          raise "#{current}: redirect without a Location header" unless target
          next
        end
        unless response.status.success?
          raise "download failed: HTTP #{code} for #{current}"
        end
        buf = Bytes.new(1 << 20)
        File.open(part, "wb") do |file|
          loop do
            n = response.body_io.read(buf)
            break if n <= 0
            file.write(buf[0, n])
            total += n
            print "\r  #{label}: #{total / 1048576} MB"
          end
        end
      end
      if target
        redirects += 1
        raise "too many redirects" if redirects > 8
        current = target.starts_with?("http") ? target : URI.parse(current).resolve(target).to_s
        next
      end
      print "\r  #{label}: #{total / 1048576} MB\n"
      File.rename(part, dest)
      return dest
    end
  end

  # ── safetensors ────────────────────────────────────────────────────────
  # The format is an 8-byte little-endian header length, that many bytes of
  # JSON describing the tensors, then the raw tensor data.
  struct Tensor
    getter dtype : String
    getter shape : Array(Int64)
    getter start : Int64
    getter stop : Int64

    def initialize(@dtype : String, @shape : Array(Int64),
                   @start : Int64, @stop : Int64)
    end
  end

  def read_tensor_header(path : String, want : String = "embeddings") : Tensor
    File.open(path, "rb") do |f|
      len_bytes = Bytes.new(8)
      f.read_fully(len_bytes)
      header_len = IO::ByteFormat::LittleEndian.decode(UInt64, len_bytes).to_i64
      header = Bytes.new(header_len)
      f.read_fully(header)
      json = JSON.parse(String.new(header))
      json.as_h.each do |name, meta|
        next unless name == want
        shape = meta["shape"].as_a.map { |v| v.as_i64 }
        offsets = meta["data_offsets"].as_a.map { |v| v.as_i64 }
        return Tensor.new(meta["dtype"].as_s, shape, offsets[0], offsets[1])
      end
      raise "#{path}: no tensor named #{want}"
    end
  end

  # ── tokenizer.json ─────────────────────────────────────────────────────
  # Pulled with a streaming parser: the multilingual tokenizer is 18 MB of
  # JSON with half a million vocabulary entries, and materialising it as a
  # tree would cost hundreds of megabytes.
  struct Vocab
    getter tokens : Array(String)
    getter scores : Array(Float32)
    getter unk_id : Int32
    getter kind : String

    def initialize(@tokens : Array(String), @scores : Array(Float32),
                   @unk_id : Int32, @kind : String)
    end
  end

  def read_vocab(path : String) : Vocab
    text = File.read(path)
    parser = JSON::PullParser.new(text)
    tokens = [] of String
    scores = [] of Float32
    unk_id = 0
    kind = ""

    parser.read_object do |key, _loc|
      if key == "model"
        parser.read_object do |mkey, _mloc|
          case mkey
          when "type"
            kind = parser.read_string
          when "unk_id"
            unk_id = parser.read_int.to_i32
          when "vocab"
            if kind == "Unigram"
              # An array of [piece, log-probability] in id order.
              parser.read_array do
                parser.read_array do
                  tokens << parser.read_string
                  scores << parser.read_float.to_f32
                end
              end
            else
              # WordPiece: an object of token -> id, which may be listed
              # out of id order.
              ids = [] of Int32
              parser.read_object do |token, _tloc|
                id = parser.read_int.to_i32
                tokens << token
                ids << id
              end
              order = Array(Int32).new(tokens.size, -1)
              tokens.each_with_index { |_, i| order[ids[i]] = i }
              sorted = order.map { |i| tokens[i] }
              tokens = sorted
              scores = [] of Float32
            end
          else
            parser.skip
          end
        end
      else
        parser.skip
      end
    end
    Vocab.new(tokens, scores, unk_id, kind)
  end

  # ── extraction ─────────────────────────────────────────────────────────
  def extract(spec : Spec, tokenizer_path : String, safetensors_path : String,
              dest : String)
    vocab = read_vocab(tokenizer_path)
    tensor = read_tensor_header(safetensors_path)
    unless tensor.dtype == "F32"
      raise "unexpected embedding dtype #{tensor.dtype} (only F32 is supported)"
    end
    v = tensor.shape[0]
    d = tensor.shape[1]
    unless vocab.tokens.size == v
      raise "vocab (#{vocab.tokens.size}) and embedding rows (#{v}) disagree"
    end

    # The magic follows the tokenizer: a "▁" prefix means SentencePiece.
    magic = spec.kind
    is_sp = vocab.tokens.any? { |t| t.includes?("▁") }
    magic = is_sp ? "M2VS" : "M2VE"
    version = is_sp ? 2 : 1

    part = dest + ".part"
    File.open(part, "wb") do |w|
      w.write(magic.to_slice)
      w.write(Semrember::Codec.u32(version))
      w.write(Semrember::Codec.u32(v.to_i32))
      w.write(Semrember::Codec.u32(d.to_i32))
      vocab.tokens.each do |t|
        b = t.to_slice
        w.write(Semrember::Codec.u16(b.size))
        w.write(b)
      end
      if version == 2
        vocab.scores.each { |s| w.write(Semrember::Codec.f32(s)) }
      end

      # Stream the embedding matrix, converting float32 to float16 on the
      # way out.  The file is never held in memory.
      File.open(safetensors_path, "rb") do |f|
        # Re-read the header length so the data offset is exact.
        len_bytes = Bytes.new(8)
        f.read_fully(len_bytes)
        header_len = IO::ByteFormat::LittleEndian.decode(UInt64, len_bytes).to_i64
        f.seek(8 + header_len + tensor.start)
        remaining = tensor.stop - tensor.start
        buf = Bytes.new(4)
        while remaining > 0
          n = {remaining, 4_i64}.min.to_i32
          f.read(buf[0, n])
          value = IO::ByteFormat::LittleEndian.decode(Float32, buf[0, n])
          w.write(Semrember::Codec.u16(Semrember::F16.from_f32(value)))
          remaining -= n
        end
      end
    end
    File.rename(part, dest)
    dest
  end

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path, "rb") do |f|
      # IO#read fills the slice and returns the count; the slice's own size
      # does not change, so the count is the loop condition.
      buf = Bytes.new(1 << 20)
      while (n = f.read(buf)) > 0
        digest.update(buf[0, n])
      end
    end
    digest.final.hexstring
  end

  # ── driver ─────────────────────────────────────────────────────────────
  def run(argv : Array(String)) : Int32
    out_dir = "models"
    force = false
    name = nil.as(String?)
    i = 0
    while i < argv.size
      case argv[i]
      when "--list"
        SPECS.each_value do |s|
          puts "#{s.name.ljust(14)} #{s.repo}  ->  #{s.file}  [#{s.kind}]"
        end
        return 0
      when "--out"  then out_dir = argv[i + 1]; i += 2
      when "--force" then force = true; i += 1
      else
        name = argv[i]
        i += 1
      end
    end
    unless (n = name)
      puts "usage: fetch_model.cr <#{SPECS.keys.join("|")}> [--out DIR] [--force]"
      puts "       fetch_model.cr --list"
      return 2
    end
    spec = SPECS[n]? || raise("unknown model #{n.inspect}; try --list")
    dest = File.join(out_dir, spec.file)

    # The whole point: if it is already there, do nothing.
    if File.exists?(dest) && !force
      got = sha256(dest)
      if got == spec.sha256
        puts "#{spec.file} is already present (#{File.size(dest) / 1048576} MB, checksum ok)"
        return 0
      end
      puts "#{spec.file} is present but its checksum does not match; refetching"
    end

    FileUtils.mkdir_p(out_dir)
    cache = File.join(out_dir, ".cache")
    tok_path = File.join(cache, "#{spec.name}-tokenizer.json")
    st_path = File.join(cache, "#{spec.name}-model.safetensors")

    unless File.exists?(tok_path)
      puts "fetching #{spec.repo}"
      download(resolve(spec.repo, "tokenizer.json"), tok_path, "tokenizer.json")
    end
    unless File.exists?(st_path)
      download(resolve(spec.repo, "model.safetensors"), st_path, "model.safetensors")
    end

    puts "extracting #{spec.file}"
    extract(spec, tok_path, st_path, dest)

    got = sha256(dest)
    if got != spec.sha256
      File.delete(dest)
      raise "checksum mismatch for #{spec.file}: expected #{spec.sha256}, got #{got}"
    end
    puts "wrote #{dest} (#{File.size(dest) / 1048576} MB, sha256 ok)"
    0
  end
end

exit FetchModel.run(ARGV)
