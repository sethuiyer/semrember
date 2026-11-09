# tokenizer.cr — WordPiece and SentencePiece-Unigram tokenizers.
#
# These implement the tokenizer paths for the M2VE and M2VS model files.
# Token IDs select embedding rows, so differences in segmentation change
# the resulting vectors.

require "./unicode"

module Semrember
  # ── WordPiece (BERT-style, "##" continuation) ─────────────────────────
  class WordPiece
    MAX_CHARS = 100

    getter vocab : Hash(String, Int32)
    getter unk_id : Int32

    def initialize(@vocab : Hash(String, Int32))
      @unk_id = @vocab["[UNK]"]? || @vocab["<unk>"]? || 0
    end

    # Split a word into punctuation pieces, the way BERT does.
    private def split_punct(word : Array(Char)) : Array(String)
      parts = [] of String
      cur = [] of Char
      word.each do |ch|
        if Unicode.ascii_punct(ch.ord) || Unicode.is_punct(ch.ord)
          unless cur.empty?
            parts << String.build { |io| cur.each { |c| io << c } }
            cur = [] of Char
          end
          parts << ch.to_s
        else
          cur << ch
        end
      end
      parts << String.build { |io| cur.each { |c| io << c } } unless cur.empty?
      parts
    end

    # Greedy longest-match over the vocabulary.
    private def piece(word : Array(Char)) : Array(Int32)
      return [@unk_id] if word.size > MAX_CHARS
      ids = [] of Int32
      start = 0
      n = word.size
      while start < n
        end_ = n
        found = nil.as(Int32?)
        while start < end_
          sub = String.build { |io| word[start, end_ - start].each { |c| io << c } }
          lookup = start == 0 ? sub : "##" + sub
          if (id = @vocab[lookup]?)
            found = id
            break
          end
          end_ -= 1
        end
        if found.nil?
          return [@unk_id]
        end
        ids << found.not_nil!
        start = end_
      end
      ids
    end

    def encode(text : String) : Array(Int32)
      ids = [] of Int32
      normalized = Unicode.bert_normalize(text)
      # Split on whitespace, then isolate punctuation inside each word.
      normalized.split(/[\s]+/).each do |word|
        next if word.empty?
        split_punct(word.chars.to_a).each do |part|
          next if part.empty?
          ids.concat(piece(part.chars.to_a))
        end
      end
      ids
    end
  end

  # ── SentencePiece Unigram ──────────────────────────────────────────────
  # With log-probabilities (M2VS v2) the segmentation is a Viterbi decode;
  # without them (v1 files) it falls back to greedy longest-match.
  class UnigramSP
    PREFIX = "▁"

    getter vocab : Hash(String, Int32)
    getter unk_id : Int32

    def initialize(@vocab : Hash(String, Int32), @scores : Array(Float32)? = nil)
      @unk_id = @vocab["[UNK]"]? || @vocab["<unk>"]? || 0
      @max_chr = 1
      @vocab.each_key { |t| @max_chr = t.chars.size if t.chars.size > @max_chr }
    end

    private def normalize(text : String) : String
      out = Unicode.sp_normalize(text)
      return "" if out.empty?
      PREFIX + out.gsub(' ', PREFIX)
    end

    private def viterbi(s : Array(Char)) : Array(Int32)
      n = s.size
      neg = -Float64::INFINITY
      best = Array(Float64).new(n + 1, neg)
      prev = Array(Int32).new(n + 1, -1)
      take = Array(Int32).new(n + 1, -1)
      best[0] = 0.0
      scores = @scores
      unk_score = scores ? scores[@unk_id].to_f64 : 0.0
      i = 0
      while i < n
        if best[i] != neg
          matched = false
          limit = {n, i + @max_chr}.min
          e = i + 1
          while e <= limit
            tok = String.build { |io| s[i, e - i].each { |c| io << c } }
            if (j = @vocab[tok]?)
              matched = true
              sc = best[i] + (scores ? scores[j].to_f64 : 0.0)
              if sc > best[e]
                best[e] = sc
                prev[e] = i
                take[e] = j
              end
            end
            e += 1
          end
          unless matched
            sc = best[i] + unk_score
            if sc > best[i + 1]
              best[i + 1] = sc
              prev[i + 1] = i
              take[i + 1] = @unk_id
            end
          end
        end
        i += 1
      end
      out = [] of Int32
      j = n
      while j > 0
        out << (take[j] >= 0 ? take[j] : @unk_id)
        j = prev[j] >= 0 ? prev[j] : j - 1
      end
      out.reverse
    end

    private def greedy(s : Array(Char)) : Array(Int32)
      out = [] of Int32
      n = s.size
      start = 0
      while start < n
        found = nil.as(Int32?)
        stop = {n, start + @max_chr}.min
        e = stop
        while e > start
          tok = String.build { |io| s[start, e - start].each { |c| io << c } }
          if (j = @vocab[tok]?)
            found = j
            break
          end
          e -= 1
        end
        out << (found || @unk_id)
        start = found ? e : start + 1
      end
      out
    end

    def encode(text : String) : Array(Int32)
      norm = normalize(text)
      return [] of Int32 if norm.empty?
      chars = norm.chars.to_a
      @scores ? viterbi(chars) : greedy(chars)
    end
  end
end
