require "./server"

# ── entry point ──────────────────────────────────────────────────────────
model = nil.as(String?)
static = nil.as(String?)
train = nil.as(String?)
key_col = "message"
val_col = "action"
port = 7379
host = "127.0.0.1"
args = ARGV.dup
i = 0
while i < args.size
  if {"--model", "--train", "--key", "--value", "--static", "--port", "--host"}.includes?(args[i]) && i + 1 >= args.size
    raise ArgumentError.new("missing value for #{args[i]}")
  end
  case args[i]
  when "--model"  then model = args[i + 1]; i += 2
  when "--train"  then train = args[i + 1]; i += 2
  when "--key"    then key_col = args[i + 1]; i += 2
  when "--value"  then val_col = args[i + 1]; i += 2
  when "--static" then static = args[i + 1]; i += 2
  when "--port"   then port = args[i + 1].to_i; i += 2
  when "--host"   then host = args[i + 1]; i += 2
  when "--help", "-h"
    puts "usage: semrember-server [--train corpus.csv --key K --value V | --model model.amem]"
    puts "                      [--static static.bin] [--host H] [--port P]"
    exit 0
  else
    raise ArgumentError.new("unknown option #{args[i]}")
  end
end
raise ArgumentError.new("port must be between 1 and 65535") unless 1 <= port <= 65535
Semrember::Server.new(model, static, train, key_col, val_col).listen(host, port.to_i32)
