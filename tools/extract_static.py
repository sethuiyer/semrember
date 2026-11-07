"""
Extract a model2vec static embedding model to the M2VE/M2VS binary format.

This is the only step that needs model2vec, and it happens once, offline.
The resulting .bin file is what the runtime reads — in Crystal, Python or C —
and it has zero runtime dependencies beyond numpy.

    pip install model2vec
    python tools/extract_static.py minishlab/potion-multilingual-128M model.bin
    python tools/extract_static.py minishlab/potion-base-8M            model.bin

The output format is documented in FORMAT.md; the encoder that reads it is
src/static_encoder.cr.

The models themselves are not redistributed here.  They come from
Minish Lab's model2vec project (MIT); see MODELS.md for the credits and the
exact sources.
"""

import sys

def extract(model_name: str, out_path: str):
    try:
        from model2vec import StaticModel
    except ImportError:
        print("pip install model2vec  (needed only for extraction, not at runtime)")
        sys.exit(1)

    import json
    import struct
    import numpy as np

    model  = StaticModel.from_pretrained(model_name)
    vocab  = model.tokenizer.get_vocab()
    id2tok = {v: k for k, v in vocab.items()}
    V, D   = model.embedding.shape
    tokens = [id2tok[i] for i in range(V)]
    E_f16  = model.embedding.astype("float16")

    # detect tokenizer: ▁ prefix = SentencePiece, ## = WordPiece
    magic = b"M2VS" if any("▁" in t for t in tokens[:500]) else b"M2VE"

    # SentencePiece Unigram files store log-probabilities (v2) so the runtime
    # can run Viterbi instead of greedy longest-match.
    scores = None
    version = 1
    if magic == b"M2VS":
        try:
            cfg = json.loads(model.tokenizer.to_str())
            vv = cfg.get("model", {}).get("vocab", [])
            if cfg.get("model", {}).get("type") == "Unigram" and len(vv) == V \
                    and all(vv[i][0] == tokens[i] for i in range(V)):
                scores = np.asarray([row[1] for row in vv], dtype="<f4")
                version = 2
        except Exception:
            scores = None

    with open(out_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<III", version, V, D))
        for tok in tokens:
            b = tok.encode("utf-8")
            f.write(struct.pack("<H", len(b))); f.write(b)
        if scores is not None:
            f.write(scores.tobytes())
        f.write(E_f16.tobytes())

    kind = "SentencePiece" if magic == b"M2VS" else "WordPiece"
    size = V * D * 2 / 1e6
    print(f"Extracted {model_name}")
    print(f"  Tokenizer : {kind} (v{version})")
    print(f"  Vocab     : {V:,}  Dim: {D}")
    print(f"  Output    : {out_path}  ({size:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
