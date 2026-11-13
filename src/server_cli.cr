require "./server"

begin
  data = nil.as(String?)
  key_file = nil.as(String?)
  model = nil.as(String?)
  static = nil.as(String?)
  train = nil.as(String?)
  key_col = "message"
  val_col = "action"
  min_counter = 0_i64
  index_enabled = false
  index_rank = 32
  index_leaf = 16
  index_seed = 0
  port = 7379
  host = "127.0.0.1"
  args = ARGV.dup
  i = 0
  while i < args.size
    if {"--model", "--train", "--key", "--value", "--static", "--port", "--host", "--data", "--encryption-key-file", "--min-counter", "--index-rank", "--index-leaf", "--index-seed"}.includes?(args[i]) && i + 1 >= args.size
      raise ArgumentError.new("missing value for #{args[i]}")
    end
    case args[i]
    when "--data"                then data = args[i + 1]; i += 2
    when "--encryption-key-file" then key_file = args[i + 1]; i += 2
    when "--min-counter"        then min_counter = args[i + 1].to_i64; i += 2
    when "--index"              then index_enabled = true; i += 1
    when "--index-rank"         then index_enabled = true; index_rank = args[i + 1].to_i; i += 2
    when "--index-leaf"         then index_enabled = true; index_leaf = args[i + 1].to_i; i += 2
    when "--index-seed"         then index_enabled = true; index_seed = args[i + 1].to_i; i += 2
    when "--model"               then model = args[i + 1]; i += 2
    when "--train"               then train = args[i + 1]; i += 2
    when "--key"                 then key_col = args[i + 1]; i += 2
    when "--value"               then val_col = args[i + 1]; i += 2
    when "--static"              then static = args[i + 1]; i += 2
    when "--port"                then port = args[i + 1].to_i; i += 2
    when "--host"                then host = args[i + 1]; i += 2
    when "--help", "-h"
      puts "usage: semrember-server [--train corpus.csv --key K --value V | --model model.amem]"
      puts "                      [--static static.bin] [--host H] [--port P]"
      puts "                      [--data state.enc --encryption-key-file secret.key]"
      puts "                      [--min-counter N]  # trusted rollback floor"
      puts "                      [--index] [--index-rank R] [--index-leaf L] [--index-seed S]"
      exit 0
    else
      raise ArgumentError.new("unknown option #{args[i]}")
    end
  end
  raise ArgumentError.new("port must be between 1 and 65535") unless 1 <= port <= 65535
  raise ArgumentError.new("min-counter must be nonnegative") if min_counter < 0
  raise ArgumentError.new("index rank must be nonnegative") if index_rank < 0
  raise ArgumentError.new("index leaf must be positive") if index_leaf <= 0
  raise ArgumentError.new("index seed must be nonnegative") if index_seed < 0
  index = index_enabled ? Semrember::Mips::Config.new(rank: index_rank, leaf_size: index_leaf, seed: index_seed) : nil
  key = key_file.try do |path|
    raise ArgumentError.new("encryption key file must contain exactly 32 bytes") unless File.size(path) == 32
    bytes = Bytes.new(32)
    begin
      File.open(path, "rb") { |file| file.read_fully(bytes) }
    rescue ex
      bytes.fill(0_u8)
      raise ex
    end
    bytes
  end
  server = nil.as(Semrember::Server?)
  begin
    server = Semrember::Server.new(model, static, train, key_col, val_col,
      data: data, encryption_key: key, min_counter: min_counter, index: index)
  ensure
    # Zero the input key whether startup succeeded or failed.
    key.try(&.fill(0_u8))
  end
  begin
    server.not_nil!.listen(host, port.to_i32)
  ensure
    server.try(&.close)
  end
rescue ex
  STDERR.puts "error: #{ex.message}"
  exit 1
end
