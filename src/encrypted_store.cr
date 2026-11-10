# Etebase-inspired key separation and authenticated encryption. Independent
# implementation of a local snapshot format, not an Etebase protocol port.
require "json"
require "file/tempfile"

@[Link("sodium")]
lib LibStoreSodium
  fun sodium_init : Int32
  fun randombytes_buf(buffer : UInt8*, length : LibC::SizeT) : Void
  fun crypto_kdf_derive_from_key(output : UInt8*, length : LibC::SizeT, id : UInt64, context : UInt8*, key : UInt8*) : Int32
  fun crypto_aead_xchacha20poly1305_ietf_encrypt(output : UInt8*, length : UInt64*, input : UInt8*, input_length : UInt64, ad : UInt8*, ad_length : UInt64, secret_nonce : UInt8*, nonce : UInt8*, key : UInt8*) : Int32
  fun crypto_aead_xchacha20poly1305_ietf_decrypt(output : UInt8*, length : UInt64*, secret_nonce : UInt8*, input : UInt8*, input_length : UInt64, ad : UInt8*, ad_length : UInt64, nonce : UInt8*, key : UInt8*) : Int32
end

module Semrember
  class PersistenceError < Exception; end

  class EncryptedStore
    MAGIC     = "SMRENC01"
    MAX_BYTES = 64 * 1024 * 1024
    @key : Bytes
    @lock_file : File
    @failed = false
    @closed = false

    def initialize(@path : String, key : Bytes)
      raise ArgumentError.new("encryption key must contain exactly 32 random bytes") unless key.size == 32
      raise PersistenceError.new("libsodium initialization failed") if LibStoreSodium.sodium_init < 0
      @key = Bytes.new(32)
      check LibStoreSodium.crypto_kdf_derive_from_key(@key, 32, 1_u64, "SemrStor".to_slice, key)
      @path = File.expand_path(@path)
      # The directory must already exist: its creation and backup policy belong to the operator.
      @lock_file = File.open(@path + ".lock", "a", perm: 0o600)
      begin
        @lock_file.flock_exclusive(blocking: false)
      rescue ex
        @lock_file.close
        raise PersistenceError.new("data file is already locked or locking is unavailable")
      end
    end

    def ensure_healthy!
      raise PersistenceError.new("persistent store unavailable; restart and inspect storage") if @failed || @closed
    end

    def load : Tuple(Hash(String, String), Hash(String, Time))
      ensure_healthy!
      return {Hash(String, String).new, Hash(String, Time).new} unless File.exists?(@path)
      raise PersistenceError.new("invalid encrypted snapshot size") unless (48..MAX_BYTES).includes?(File.size(@path))
      wire = File.read(@path).to_slice
      raise PersistenceError.new("unsupported snapshot format") unless wire[0, 8] == MAGIC.to_slice
      plain = Bytes.new(wire.size - 48)
      length = 0_u64
      check LibStoreSodium.crypto_aead_xchacha20poly1305_ietf_decrypt(plain, pointerof(length), Pointer(UInt8).null,
        wire[32..], wire.size - 32, wire[0, 8], 8, wire[8, 24], @key)
      parsed = JSON.parse(String.new(plain))
      raise PersistenceError.new("unsupported snapshot payload version") unless parsed["version"].as_i == 1
      records = {} of String => String
      parsed["records"].as_h.each { |key, value| records[key] = value.as_s }
      expiry = {} of String => Time
      parsed["expiry"].as_h.each do |key, value|
        raise PersistenceError.new("expiry without a record") unless records.has_key?(key)
        expiry[key] = Time.unix_ms(value.as_i64)
      end
      {records, expiry}
    end

    def save(records : Hash(String, String), expiry : Hash(String, Time)) : Nil
      ensure_healthy!
      payload = {version: 1, records: records, expiry: expiry.transform_values(&.to_unix_ms)}.to_json.to_slice
      raise PersistenceError.new("snapshot exceeds 64 MiB limit") if payload.size > MAX_BYTES - 48
      nonce = Bytes.new(24)
      LibStoreSodium.randombytes_buf(nonce, 24)
      encrypted = Bytes.new(payload.size + 16)
      length = 0_u64
      check LibStoreSodium.crypto_aead_xchacha20poly1305_ietf_encrypt(encrypted, pointerof(length), payload,
        payload.size, MAGIC.to_slice, 8, Pointer(UInt8).null, nonce, @key)
      temp = File.tempfile(".semrember-", ".tmp", dir: File.dirname(@path))
      begin
        temp.chmod(0o600)
        temp.write(MAGIC.to_slice)
        temp.write(nonce)
        temp.write(encrypted)
        temp.flush
        temp.fsync
        temp.close
        File.rename(temp.path, @path)
        File.open(File.dirname(@path)) { |directory| directory.fsync }
      rescue ex
        # A failure after rename has an uncertain commit outcome. Never continue
        # serving stale memory; require recovery from the authenticated snapshot.
        @failed = true
        raise PersistenceError.new("snapshot commit failed; restart and inspect storage")
      ensure
        temp.close unless temp.closed?
        File.delete(temp.path) if File.exists?(temp.path)
      end
    end

    def close
      return if @closed
      @closed = true
      @key.fill(0_u8)
      @lock_file.close
    end

    private def check(status : Int32)
      raise PersistenceError.new("encrypted snapshot authentication or crypto operation failed") unless status == 0
    end
  end
end
