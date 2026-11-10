# Encrypted durable storage

The HTTP server can persist its exact records and absolute TTL deadlines. On
restart it authenticates/decrypts the snapshot, removes expired records, and
rebuilds a writable dual-solver semantic index. Default mode remains in-memory.

## Start

Build with libsodium installed (`libsodium-dev` on Debian/Ubuntu):

```sh
make server
mkdir -m 700 private-state
(umask 077; openssl rand -out private-state/secret.key 32)
build/semrember-server --data private-state/state.enc \
  --encryption-key-file private-state/secret.key
```

Generate the key **once**, retain it securely, and reuse both paths on restart.
The key is exactly 32 cryptographically random binary bytes, not a password or
hex string. Keep key backups separately from encrypted snapshot backups. Losing
the key loses access to the data. Key rotation is not implemented in this version.
Protect the directory from other writers; do not delete the `.lock` file while a
process is running. A second cooperating process using the same path is refused.
Use a local filesystem that supports atomic rename, flock and directory fsync.

`--data` and `--encryption-key-file` must be supplied together. They cannot be
combined with `--model` or `--train`. `--static` is supported. When a static
encoder is used, the snapshot records its tokenizer, dimension and a SHA-256
fingerprint. Restarting with a different encoder is rejected instead of silently
changing semantic scores. A lexical snapshot can be upgraded by restarting with
`--static`; the encoder identity is recorded then. Encoder binaries are not
included in snapshots. `/v1/info` reports `persistence: encrypted-snapshot`
or `persistence: memory`.

## Commit and recovery

Every PUT, DELETE, successful EXPIRE and expiration purge builds candidate records
and deadlines before changing live state. The semantic index is rebuilt only when
records change; an EXPIRE or same-value PUT updates the snapshot without refitting.
On restart, a missing snapshot is created, but an existing valid snapshot is not
rewritten unless expired records were purged or encoder metadata is being recorded.
A mutation then:

1. Encrypts the entire snapshot using a fresh 24-byte nonce.
2. Writes a mode-0600 temporary file in the snapshot directory and fsyncs it.
3. Atomically renames it over the snapshot and fsyncs the containing directory.
4. Publishes the candidate state and acknowledges the operation.

A successful response means the filesystem has acknowledged the durable commit.
Termination before acknowledgement can leave either the old or new complete
snapshot. A failure during replacement poisons the store: subsequent data
requests fail until restart rather than returning uncertain state. Wrong keys,
changed ciphertext and truncated files fail startup, without resetting the store.
Uncommitted temporary files are never loaded; a hard kill may leave encrypted
temporary files that can be removed while the service is stopped.

Snapshots contain all exact keys, values and UTC deadlines. Deletion removes
records from the next snapshot; it does not securely erase old filesystem blocks
or backups. Expiration remains lazy during operation and is also applied at boot.

## Encryption and relationship to EteSync

This uses Etebase's core cryptographic pattern: libsodium BLAKE2b KDF domain
separation and XChaCha20-Poly1305 authenticated encryption. The 32-byte input key
derives a 32-byte cipher key using subkey ID 1 and the eight-byte context
`SemrStor`. The snapshot header is authenticated as additional data.

It is an independent local format, **not an Etebase-compatible sync client**.
It does not import the unfinished EteSync port or its source-derived code.
Chunk shuffling, content deduplication, multi-device sync and conflict resolution
are unnecessary for this first whole-snapshot implementation and are not included.
The compatibility spec decrypts a fixture produced by the official Etebase JS
CryptoManager (`etebase@0.43.0`) with this context and additional data.

Wire format: `SMRENC01` (8 bytes) || nonce (24 bytes) || ciphertext || tag (16 bytes).
Plaintext: JSON `{version: 2, encoder: null | "Tokenizer:dim:sha256", records:
{key: value}, expiry: {key: unix_ms}}`. Version 1 snapshots are accepted and
upgraded on the next successful write. The total snapshot size is limited to
64 MiB. Unsupported versions are rejected.

References: [Etebase crypto primitives](https://docs.etebase.com/protocol-specs/introduction),
[upstream crypto implementation](https://github.com/etesync/etebase-rs/blob/c0c74b777724777713d0fbbaeb968ba1521aa8e8/src/crypto.rs).

## Limits and cost

This protects data at rest. The running process holds plaintext and the key; HTTP
authentication/TLS are separate deployment concerns. Snapshot sizes are visible.
An attacker replacing the snapshot with an older valid snapshot, deleting it,
or replacing it with another valid snapshot encrypted using the same key is not
detected. Use a distinct key per store and trusted backups/checkpoints if you need
rollback detection. A missing data file initializes a new empty store.

Each mutation rewrites the entire snapshot while holding the service lock; record
changes also refit the index. This trades throughput and temporary memory for a
simple atomic commit boundary and suits small stores. Refit updates vocabulary/IDF,
so semantic scores can differ from default incremental mode. EXPIRE and same-value
PUTs still rewrite the snapshot, but reuse the current semantic index. No WAL, replication, incremental
checkpointing or model-weight snapshot is claimed. AMEM export remains plaintext
and unchanged; use this mode for the service's durable records.
