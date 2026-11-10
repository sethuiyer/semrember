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
    MAGIC          = "SMRENC01"
    FORMAT_VERSION = 2
    MAX_BYTES      = 64 * 1024 * 1024

    getter encoder_upgrade_pending : Bool

    @key : Bytes
    @lock_file : File
    @failed = false
    @closed = false

    def initialize(@path : String, key : Bytes, @encoder_id : String? = nil)
      raise ArgumentError.new("encryption key must contain exactly 32 random bytes") unless key.size == 32
      raise PersistenceError.new("libsodium initialization failed") if LibStoreSodium.sodium_init < 0
      @key = Bytes.new(32)
      @encoder_upgrade_pending = false
      begin
        check LibStoreSodium.crypto_kdf_derive_from_key(@key, 32, 1_u64, "SemrStor".to_slice, key)
      rescue ex
        @key.fill(0_u8)
        raise ex
      end
      @path = File.expand_path(@path)
      # The directory must already exist: its creation and backup policy belong to the operator.
      lock = nil.as(File?)
      begin
        lock = File.open(@path + ".lock", "a", perm: 0o600)
      rescue ex
        @key.fill(0_u8)
        raise ex
      end
      begin
        lock.not_nil!.flock_exclusive(blocking: false)
      rescue ex
        @key.fill(0_u8)
        lock.not_nil!.close
        raise PersistenceError.new("data file is already locked or locking is unavailable")
      end
      @lock_file = lock.not_nil!
    end

    def exists? : Bool
      File.exists?(@path)
    end

    def ensure_healthy!
      raise PersistenceError.new("persistent store unavailable; restart and inspect storage") if @failed || @closed
    end

    def load : Tuple(Hash(String, String), Hash(String, Time))
      ensure_healthy!
      @encoder_upgrade_pending = false
      return {Hash(String, String).new, Hash(String, Time).new} unless File.exists?(@path)

      # Read through a single open descriptor and read exactly the size that was
      # checked. A concurrent replacement cannot make us allocate an unbounded
      # buffer between the size check and the read.
      wire = File.open(@path, "rb") do |file|
        size = file.size
        unless size >= 48 && size <= MAX_BYTES
          raise PersistenceError.new("invalid encrypted snapshot size")
        end
        bytes = Bytes.new(size.to_i32)
        begin
          file.read_fully(bytes)
        rescue IO::EOFError
          raise PersistenceError.new("invalid encrypted snapshot size")
        end
        raise PersistenceError.new("snapshot changed while reading") unless file.size == size
        bytes
      end

      raise PersistenceError.new("unsupported snapshot format") unless wire[0, 8] == MAGIC.to_slice
      plain = Bytes.new(wire.size - 48)
      length = 0_u64
      check LibStoreSodium.crypto_aead_xchacha20poly1305_ietf_decrypt(plain, pointerof(length), Pointer(UInt8).null,
        wire[32..], wire.size - 32, wire[0, 8], 8, wire[8, 24], @key)

      object = begin
        JSON.parse(String.new(plain))
      rescue ex : JSON::ParseException
        raise PersistenceError.new("invalid snapshot payload")
      end
      version = object["version"].as_i
      raise PersistenceError.new("unsupported snapshot payload version") unless version == 1 || version == 2
      hash = object.as_h
      has_encoder = hash.has_key?("encoder")
      raise PersistenceError.new("snapshot is missing encoder metadata") if version >= 2 && !has_encoder
      if has_encoder
        snapshot_encoder = hash["encoder"].as_s?
        if snapshot_encoder
          unless snapshot_encoder == @encoder_id
            raise PersistenceError.new("snapshot was written with a different static encoder; restart with the original encoder")
          end
        else
          # A lexical snapshot can be upgraded by starting with --static. The
          # encoder identity is recorded by the next successful save.
          @encoder_upgrade_pending = !@encoder_id.nil?
        end
      else
        # Version 1 snapshots predate encoder metadata.
        @encoder_upgrade_pending = !@encoder_id.nil?
      end

      records = {} of String => String
      hash["records"].as_h.each { |key, value| records[key] = value.as_s }
      expiry = {} of String => Time
      hash["expiry"].as_h.each do |key, value|
        raise PersistenceError.new("expiry without a record") unless records.has_key?(key)
        expiry[key] = Time.unix_ms(value.as_i64)
      end
      {records, expiry}
    rescue ex : JSON::ParseException | TypeCastError | KeyError | ArgumentError | OverflowError | IO::Error
      raise PersistenceError.new("invalid snapshot payload")
    end

    def save(records : Hash(String, String), expiry : Hash(String, Time)) : Nil
      ensure_healthy!
      json = {
        version: FORMAT_VERSION,
        encoder: @encoder_id,
        records: records,
        expiry:  expiry.transform_values(&.to_unix_ms),
      }.to_json
      payload = json.to_slice
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
        @encoder_upgrade_pending = false
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
