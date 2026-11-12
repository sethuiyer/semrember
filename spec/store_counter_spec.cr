require "spec"
require "file_utils"
require "../src/encrypted_store"

private def with_counter_store(&block : String, Bytes ->)
  dir = File.tempname("semrember-counter")
  Dir.mkdir(dir, 0o700)
  begin
    yield File.join(dir, "state.enc"), Bytes.new(32, 23_u8)
  ensure
    FileUtils.rm_rf(dir)
  end
end

describe Semrember::EncryptedStore do
  it "increments the counter and rejects a snapshot below the trusted floor" do
    with_counter_store do |path, key|
      store = Semrember::EncryptedStore.new(path, key)
      store.save({"a" => "b"}, {} of String => Time)
      store.counter.should eq 1
      store.close

      store = Semrember::EncryptedStore.new(path, key, min_counter: 1)
      store.load[0]["a"].should eq "b"
      store.counter.should eq 1
      store.save({"a" => "c"}, {} of String => Time)
      store.counter.should eq 2
      store.close

      refused = Semrember::EncryptedStore.new(path, key, min_counter: 3)
      begin
        expect_raises(Semrember::PersistenceError, "snapshot rollback detected") { refused.load }
      ensure
        refused.close
      end
    end
  end

  it "detects a replayed older snapshot" do
    with_counter_store do |path, key|
      store = Semrember::EncryptedStore.new(path, key)
      store.save({"a" => "old"}, {} of String => Time)
      old = File.read(path)
      store.save({"a" => "new"}, {} of String => Time)
      store.close

      File.write(path, old)
      replayed = Semrember::EncryptedStore.new(path, key, min_counter: 2)
      begin
        expect_raises(Semrember::PersistenceError, "snapshot rollback detected") { replayed.load }
      ensure
        replayed.close
      end
    end
  end

  it "starts a missing snapshot above the trusted floor" do
    with_counter_store do |path, key|
      store = Semrember::EncryptedStore.new(path, key, min_counter: 5)
      store.load[0].should be_empty
      store.counter.should eq 5
      store.save({"a" => "b"}, {} of String => Time)
      store.counter.should eq 6
      store.close
    end
  end
end
