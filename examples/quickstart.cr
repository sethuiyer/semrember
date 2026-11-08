# From the repository root: crystal run examples/quickstart.cr
# This uses the engine directly: no HTTP service or Kemal is required.
require "../src/semrember"

memory = Semrember::Memory.new.fit({
  "disk full"               => "free_disk",
  "certificate has expired" => "renew_certificate",
})
raise "unexpected label" unless memory.query("certificate has expired")[0][0] == "renew_certificate"
puts "Learned label: #{memory.query("certificate has expired")[0][0]}"
raise "wrong previous value" unless memory.set("disk full", "clear_cache") == "free_disk"

path = File.tempname("semrember-quickstart", ".amem")
begin
  memory.save(path)
  reader = Semrember::Memory.load(path)
  raise "unexpected saved label" unless reader.query("disk full")[0][0] == "clear_cache"
  puts "Saved reader predicts: #{reader.query("disk full")[0][0]}"
  begin
    reader.set("disk full", "free_disk")
    raise "loaded models must reject writes"
  rescue ex : Semrember::UnavailableError
    puts "Loaded AMEM is read-only; refit from source records to resume updates."
  end
ensure
  File.delete(path) if File.exists?(path)
end
