require "./server_spec"
require "file_utils"

private def with_store(&block : String, Bytes ->)
  dir = File.tempname("semrember-durable")
  Dir.mkdir(dir, 0o700)
  begin
    yield File.join(dir, "state.enc"), Bytes.new(32, 42_u8)
  ensure
    FileUtils.rm_rf(dir)
  end
end

# Two tiny M2VE models with the same dimension but different bytes.
private def write_test_encoder(path : String, variant : Int32 = 0, dim : Int32 = 2)
  File.open(path, "wb") do |file|
    file << "M2VE"
    [1, 3, dim].each { |value| file.write(Semrember::Codec.u32(value)) }
    ["[UNK]", "disk", "host"].each do |token|
      file.write(Semrember::Codec.u16(token.bytesize))
      file << token
    end
    3.times do |i|
      dim.times do |d|
        value = d == (i + variant) % dim ? 1_f32 : 0_f32
        file.write(Semrember::Codec.u16(Semrember::F16.from_f32(value)))
      end
    end
  end
end

private def with_test_encoders(&block : String, String ->)
  dir = File.tempname("semrember-encoders")
  Dir.mkdir(dir, 0o700)
  begin
    one = File.join(dir, "one.bin")
    two = File.join(dir, "two.bin")
    write_test_encoder(one, variant: 0)
    write_test_encoder(two, variant: 1)
    yield one, two
  ensure
    FileUtils.rm_rf(dir)
  end
end

describe Semrember::EncryptedStore do
  it "reads a fixture encrypted with the official Etebase JS CryptoManager" do
    with_store do |path, key|
      # etebase@0.43.0, key=42*32, context=SemrStor, AD=SMRENC01.
      hex = "534d52454e4330318ad370f9eb6e2b11eb0a7937f866a0670981d426ba2df1b15ec21d5fd99645786a1e7851cf06f19e49d00061c485f8e298f9415d2e59114d127390036e8fb2003743445a114547248a519c0fff00d29e2a08d251d8c2647f29efe4f651b5e9dd99b6fe396a"
      File.write(path, hex.hexbytes)
      store = Semrember::EncryptedStore.new(path, key)
      begin
        store.load[0].should eq({"disk full" => "free_disk"})
      ensure
        store.close
      end
    end
  end

  it "uses fresh nonces, private files, and authenticates every byte" do
    with_store do |path, key|
      store = Semrember::EncryptedStore.new(path, key)
      begin
        records = {"very secret key" => "very secret value"}
        store.save(records, {} of String => Time)
        first = File.read(path)
        first.includes?("very secret").should be_false
        (File.info(path).permissions.value & 0o777).should eq(0o600)
        store.save(records, {} of String => Time)
        File.read(path).should_not eq(first)
        wire = first.to_slice.dup
        wire.size.times do |i|
          corrupted = wire.dup
          corrupted[i] ^= 1_u8
          File.write(path, corrupted)
          expect_raises(Semrember::PersistenceError) { store.load }
        end
        [0, 7, 31, 47, wire.size - 1].each do |size|
          File.write(path, wire[0, size])
          expect_raises(Semrember::PersistenceError) { store.load }
        end
      ensure
        store.close
      end
    end
  end

  it "rejects a second writer and releases the lock on close" do
    with_store do |path, key|
      first = Semrember::EncryptedStore.new(path, key)
      begin
        expect_raises(Semrember::PersistenceError) { Semrember::EncryptedStore.new(path, key) }
      ensure
        first.close
      end
      Semrember::EncryptedStore.new(path, key).close
    end
  end
end

describe "Encrypted durable service" do
  it "does not rewrite a valid snapshot when a restart has nothing to persist" do
    with_store do |path, key|
      server = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        put(server, "disk", "free_disk")[0].should eq(200)
      ensure
        server.close
      end
      before = File.read(path)
      restarted = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        File.read(path).should eq before
      ensure
        restarted.close
      end
    end
  end

  it "records a static encoder identity and rejects a different encoder on restart" do
    with_store do |path, key|
      with_test_encoders do |one, two|
        server = Semrember::Server.new(nil, one, data: path, encryption_key: key)
        begin
          put(server, "disk", "free_disk")[0].should eq(200)
        ensure
          server.close
        end
        expect_raises(Semrember::PersistenceError) do
          Semrember::Server.new(nil, two, data: path, encryption_key: key)
        end
        expect_raises(Semrember::PersistenceError) do
          Semrember::Server.new(nil, nil, data: path, encryption_key: key)
        end
        Semrember::Server.new(nil, one, data: path, encryption_key: key).close
      end
    end
  end

  it "allows upgrading a lexical snapshot to a static encoder, then locks it" do
    with_store do |path, key|
      with_test_encoders do |one, two|
        lexical = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
        begin
          put(lexical, "disk", "free_disk")[0].should eq(200)
        ensure
          lexical.close
        end
        Semrember::Server.new(nil, one, data: path, encryption_key: key).close
        expect_raises(Semrember::PersistenceError) do
          Semrember::Server.new(nil, two, data: path, encryption_key: key)
        end
        Semrember::Server.new(nil, one, data: path, encryption_key: key).close
      end
    end
  end

  it "restores exact records, semantic reads, deletes, TTLs and writable state" do
    with_store do |path, key|
      server = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        put(server, "disk full", "free_disk")[0].should eq(200)
        put(server, "temporary", "remove_me")[0].should eq(200)
        request(server, "DELETE", "/v1/kv/temporary")[0].should eq(200)
        request(server, "POST", "/v1/kv/disk%20full/expire", {seconds: 3600}.to_json)[0].should eq(200)
      ensure
        server.close
      end
      restarted = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        request(restarted, "GET", "/v1/kv/disk%20full")[1]["value"].as_s.should eq("free_disk")
        request(restarted, "GET", "/v1/kv/temporary")[0].should eq(404)
        request(restarted, "GET", "/v1/kv/disk%20full/ttl")[1]["ttl"].as_i.should be > 3500
        result = request(restarted, "POST", "/v1/similar", {text: "disk full"}.to_json)
        result[0].should eq(200)
        result[1]["candidates"][0]["key"].as_s.should eq("disk full")
        put(restarted, "disk full", "clear_cache")[0].should eq(200)
        request(restarted, "GET", "/v1/kv/disk%20full/ttl")[1]["ttl"].raw.should be_nil
      ensure
        restarted.close
      end
    end
  end

  it "drops records whose absolute TTL elapsed while offline" do
    with_store do |path, key|
      store = Semrember::EncryptedStore.new(path, key)
      store.save({"expired" => "secret", "keep" => "value"}, {"expired" => Time.utc - 1.second})
      store.close
      server = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        request(server, "GET", "/v1/kv/expired")[0].should eq(404)
        server.records.should eq({"keep" => "value"})
      ensure
        server.close
      end
    end
  end

  it "fails startup on a wrong key without overwriting data or retaining the lock" do
    with_store do |path, key|
      Semrember::Server.new(nil, nil, data: path, encryption_key: key).close
      before = File.read(path)
      expect_raises(Semrember::PersistenceError) do
        Semrember::Server.new(nil, nil, data: path, encryption_key: Bytes.new(32, 1_u8))
      end
      File.read(path).should eq(before)
      Semrember::Server.new(nil, nil, data: path, encryption_key: key).close
    end
  end

  it "does not publish a failed commit and refuses further requests" do
    with_store do |path, key|
      server = Semrember::Server.new(nil, nil, data: path, encryption_key: key)
      begin
        put(server, "disk", "old")[0].should eq(200)
        File.rename(path, path + ".backup")
        Dir.mkdir(path) # Force atomic rename to fail.
        put(server, "disk", "new")[0].should eq(500)
        server.records["disk"].should eq("old")
        request(server, "GET", "/v1/kv/disk")[0].should eq(500)
        Dir.glob(File.join(File.dirname(path), ".semrember-*.tmp")).should be_empty
      ensure
        server.close
      end
    end
  end

  it "rejects ambiguous startup configurations" do
    expect_raises(ArgumentError) { Semrember::Server.new(nil, nil, data: "unused") }
    expect_raises(ArgumentError) { Semrember::Server.new(nil, nil, encryption_key: Bytes.new(32)) }
    expect_raises(ArgumentError) { Semrember::Server.new("unused", nil, data: "unused", encryption_key: Bytes.new(32)) }
    with_store do |path, key|
      expect_raises(ArgumentError) { Semrember::Server.new(nil, nil, data: path, encryption_key: key[0, 31]) }
    end
  end
end
