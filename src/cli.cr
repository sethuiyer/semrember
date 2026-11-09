# cli.cr — command line front end.
#
#   semrember train  <data.csv> <key_col> <val_col> <out.amem> [min_df] [solver]
#   semrember query  <model.amem> <text> [topk]
#   semrember scores <model.amem> <text>
#   semrember info   <model.amem>
#
# Every command that prints machine-readable output emits JSON on stdout, so
# the Crystal and Python implementations can be diffed byte for byte.

require "json"
require "./semrember"
require "./csv"

module Semrember
  module Cli
    extend self

    private def usage(io : IO)
      io.puts "Usage:"
      io.puts "  semrember train  <data.csv> <key_col> <val_col> <out.amem> [min_df] [solver]"
      io.puts "  semrember query  <model.amem> <text> [topk]"
      io.puts "  semrember scores <model.amem> <text>"
      io.puts "  semrember info   <model.amem>"
    end

    private def load_csv(path : String, key_col : String, val_col : String) : Hash(String, String)
      data = {} of String => String
      Csv.load_pairs(path, key_col, val_col).each { |(k, v)| data[k] = v }
      data
    end

    # `--static <path>` may appear anywhere; it is stripped before the
    # command-specific arguments are parsed.
    private def extract_static(argv : Array(String)) : Tuple(Array(String), String?)
      args = [] of String
      static = nil.as(String?)
      i = 0
      while i < argv.size
        if argv[i] == "--static" && i + 1 < argv.size
          static = argv[i + 1]
          i += 2
          next
        end
        if argv[i].starts_with?("--static=")
          static = argv[i].split("=", 2)[1]
          i += 1
          next
        end
        args << argv[i]
        i += 1
      end
      {args, static}
    end

    def run(argv : Array(String)) : Int32
      argv, static_path = extract_static(argv)
      if argv.empty? || {"-h", "--help", "help"}.includes?(argv[0])
        usage(STDOUT)
        return 0
      end
      static = static_path.try { |p| StaticEncoder.new(p) }
      case argv[0]
      when "train"
        raise "usage: train <data.csv> <key_col> <val_col> <out.amem> [min_df] [solver]" if argv.size < 5
        data = load_csv(argv[1], argv[2], argv[3])
        out_path = argv[4]
        min_df = argv.size > 5 ? argv[5].to_i : 1
        solver = :dual
        if argv.size > 6
          solver = case argv[6]
                   when "delta" then :delta
                   when "rls"   then :rls
                   when "dual"  then :dual
                   else              raise ArgumentError.new("solver must be delta, rls or dual")
                   end
        end
        mem = Memory.new(solver: solver, min_df: min_df, static: static)
        mem.fit(data)
        mem.save(out_path)
        puts JSON.build { |j|
          j.object do
            j.field "trained", data.size
            j.field "labels", mem.labels.size
            j.field "dim", mem.d_tfidf + mem.d_static
          end
        }
        0
      when "query"
        raise "usage: query <model.amem> <text> [topk]" if argv.size < 3
        mem = Memory.load(argv[1], static: static)
        topk = argv.size > 3 ? argv[3].to_i : 1
        hits = mem.query(argv[2], topk)
        puts JSON.build { |j|
          j.object do
            j.field "hits" do
              j.array do
                hits.each do |(label, score)|
                  j.object do
                    j.field "label", label
                    j.field "score", score
                  end
                end
              end
            end
          end
        }
        0
      when "scores"
        raise "usage: scores <model.amem> <text>" if argv.size < 3
        mem = Memory.load(argv[1], static: static)
        s = mem.scores(argv[2])
        puts JSON.build { |j|
          j.object do
            j.field "labels" do
              j.array { mem.labels.each { |l| j.string l } }
            end
            j.field "scores" do
              j.array { s.each { |v| j.number v } }
            end
          end
        }
        0
      when "encode"
        raise "usage: encode <model.amem> <text>" if argv.size < 3
        mem = Memory.load(argv[1], static: static)
        f = mem.encode(argv[2])
        puts JSON.build { |j|
          j.object do
            j.field "idx" do
              j.array { f.idx.each { |v| j.number v } }
            end
            j.field "val" do
              j.array { f.val.each { |v| j.number v } }
            end
          end
        }
        0
      when "dense"
        raise "usage: dense <model.amem> <text>" if argv.size < 3
        mem = Memory.load(argv[1], static: static)
        f = mem.encode(argv[2])
        pairs = f.idx.each_with_index.map { |i, k| {i, f.val[k]} }
        puts JSON.build { |j|
          j.object do
            j.field "st" do
              j.array { f.st.each { |v| j.number v } }
            end
            j.field "pairs" do
              j.array do
                pairs.each do |(i, v)|
                  j.object do
                    j.field "i", i
                    j.field "v", v
                  end
                end
              end
            end
          end
        }
        0
      when "tokens"
        raise "usage: tokens <static.bin> <text>" if argv.size < 3
        enc = StaticEncoder.new(argv[1])
        ids = enc.encode_ids(argv[2])
        puts JSON.build { |j| j.array { ids.each { |v| j.number v } } }
        0
      when "info"
        raise "usage: info <model.amem>" if argv.size < 2
        mem = Memory.load(argv[1], static: static)
        puts JSON.build { |j|
          j.object do
            j.field "labels" do
              j.array { mem.labels.each { |l| j.string l } }
            end
            j.field "d_tfidf", mem.d_tfidf
            j.field "d_static", mem.d_static
            j.field "min_df", mem.min_df
            j.field "vocab_size", mem.vocab_size
          end
        }
        0
      else
        usage(STDERR)
        1
      end
    end
  end
end

begin
  exit Semrember::Cli.run(ARGV)
rescue ex
  STDERR.puts "error: #{ex.message}"
  exit 1
end
