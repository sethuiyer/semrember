# static_encoder.cr — zero-dependency static embedding encoder.
#
# Loads a pre-extracted M2VE (WordPiece) or M2VS (SentencePiece Unigram)
# model and embeds text as a mean-pooled, L2-normalised vector.
#
# The C implementation memory-maps the file; Crystal has no mmap binding
# without libclang, so the file is read once into memory.  For the 15 MB
# base model that is irrelevant; for the 265 MB multilingual model it costs
# resident memory.  The lexical-only path — the "floppy" mode — needs no
# encoder at all.

require "digest/sha256"
require "./codec"
require "./tokenizer"

module Semrember
  class StaticEncoder
    MAGIC_WP = "M2VE"
    MAGIC_SP = "M2VS"

    getter dim : Int32
    getter vocab_size : Int32
    getter tokenizer : String

    @data : Bytes
    @vocab : Hash(String, Int32)
    @scores : Array(Float32)?
    @tok : WordPiece | UnigramSP
    @pool_unk_id : Int32?
    @fingerprint : String?

    def initialize(path : String)
      @data = File.read(path).to_slice
      if @data.size < 16
        raise "#{path}: truncated model"
      end
      magic = String.new(@data[0, 4])
      unless magic == MAGIC_WP || magic == MAGIC_SP
        raise "#{path}: unrecognised magic #{magic.inspect}"
      end
      off = 4
      version = Codec.read_u32(@data, off).to_i32; off += 4
      v = Codec.read_u32(@data, off).to_i32; off += 4
      d = Codec.read_u32(@data, off).to_i32; off += 4
      if version < 1 || version > 2 || (magic == MAGIC_WP && version != 1)
        raise "#{path}: unsupported version #{version}"
      end
      unless v > 0 && d > 0 && d <= 65536
        raise "#{path}: bad dimensions V=#{v} D=#{d}"
      end
      @dim = d
      @vocab_size = v
      vocab = {} of String => Int32
      v.times do |i|
        len = Codec.read_u16(@data, off).to_i32; off += 2
        vocab[String.new(@data[off, len])] = i
        off += len
      end
      @vocab = vocab
      scores = nil.as(Array(Float32)?)
      if magic == MAGIC_SP && version >= 2
        raw = @data[off, v * 4]
        if raw.size != v * 4
          raise "#{path}: truncated score table"
        end
        scores = Array(Float32).new(v, 0_f32)
        v.times { |i| scores[i] = Codec.read_f32(@data, off + i * 4) }
        off += v * 4
      end
      @scores = scores
      @embedding_offset = off
      need = off + v * d * 2
      raise "#{path}: truncated embedding matrix" if @data.size < need
      @embedding_count = v
      @tok = if magic == MAGIC_WP
               WordPiece.new(vocab)
             else
               UnigramSP.new(vocab, scores)
             end
      @pool_unk_id = @tok.as?(WordPiece).try(&.unk_id)
      @tokenizer = magic == MAGIC_WP ? "WordPiece" : "SentencePiece"
    end

    # Computed lazily: only durable mode needs an encoder identity, and hashing a
    # 265 MB model should not slow down every CLI query that loads it.
    def fingerprint : String
      @fingerprint ||= Digest::SHA256.hexdigest(@data)
    end

    @embedding_offset : Int32
    @embedding_count : Int32

    private def embedding(cp_id : Int32, d : Int32) : Float32
      off = @embedding_offset + (cp_id.to_i64 * @dim.to_i64 + d.to_i64) * 2
      F16.to_f32(Codec.read_u16(@data, off.to_i32))
    end

    # The reference implementation keeps this vector in float64 until the
    # caller casts it, and the joint norm is computed over that float64
    # value.  Matching that matters: the float32 variant rounds the norm
    # early and the difference propagates into every weight.
    #
    # Two details of the reference arithmetic are reproduced here:
    #   * the token mean is a sequential float32 accumulation (numpy uses
    #     float32 for a strided axis-0 reduction, and for the handful of
    #     tokens in a typical query that is a plain sequential sum);
    #   * the norm's sum of squares is accumulated in float64, which lands
    #     closer to numpy's BLAS dot product than a float32 sum would.
    def encode64(text : String) : Array(Float64)
      ids = encode_ids(text)
      if unk_id = @pool_unk_id
        ids.reject! { |id| id == unk_id }
      end
      return [] of Float64 if ids.empty?

      vec = Array(Float32).new(@dim, 0_f32)
      ids.each do |id|
        @dim.times { |d| vec[d] = (vec[d] + embedding(id, d)).to_f32 }
      end
      count = ids.size.to_f32
      vec.map! { |v| v / count }
      s = 0.0_f64
      vec.each { |v| s += v.to_f64 * v.to_f64 }
      norm = Math.sqrt(s) + 1e-9
      vec.map { |v| v.to_f64 / norm }
    end

    def encode(text : String) : Array(Float32)
      encode64(text).map { |v| v.to_f32 }
    end

    # Raw tokenizer IDs, including unknown pieces before pooling.
    def encode_ids(text : String) : Array(Int32)
      @tok.encode(text)
    end

    def close
      @data = Bytes.empty
    end
  end
end
