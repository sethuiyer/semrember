# unicode.cr — Unicode predicates and normalization shared with the C library.
#
# The tables in unicode_tables.cr are generated from the C implementation, so
# the two agree on lowercasing, mark stripping and punctuation by construction
# rather than by coincidence.
#
# One subtlety is worth stating, because it is the kind of thing that silently
# breaks tokenizer parity: Unicode lowercasing is *contextual*.  Python's
# `str.lower()` maps "ΣΣ" to "σς" — the second sigma is final and becomes the
# final-sigma form — while Crystal's `String#downcase` maps each codepoint
# independently and produces "σσ".  The final-sigma rule is implemented here
# explicitly, using the cased and case-ignorable ranges, exactly as the C
# library does.

require "./unicode_tables"

module Semrember
  module Unicode
    extend self

    # ── table lookups ──────────────────────────────────────────────────
    private def in_ranges(cp : Int32, ranges : Array(Tuple(UInt32, UInt32))) : Bool
      c = cp.to_u32
      lo = 0
      hi = ranges.size
      while lo < hi
        mid = (lo + hi) // 2
        r = ranges[mid]
        if c < r[0]
          hi = mid
        elsif c > r[1]
          lo = mid + 1
        else
          return true
        end
      end
      false
    end

    # Full lowercase mapping; may yield up to three codepoints.
    def lower_cp(cp : Int32) : Array(UInt32)
      LOWER[cp.to_u32]?.try(&.dup) || [cp.to_u32]
    end

    def is_mn(cp : Int32) : Bool
      in_ranges(cp, MN_RANGES)
    end

    def is_punct(cp : Int32) : Bool
      in_ranges(cp, PUNCT_RANGES)
    end

    def is_cased(cp : Int32) : Bool
      in_ranges(cp, CASED_RANGES)
    end

    def is_case_ignorable(cp : Int32) : Bool
      in_ranges(cp, CASE_IGNORABLE_RANGES)
    end

    # Matches Python's `str.isspace()` exactly — the same set the C library
    # uses, which is what SentencePiece's normalizer expects.
    def is_space(cp : Int32) : Bool
      cp == 0x20 || (cp >= 0x09 && cp <= 0x0D) ||
        (cp >= 0x1C && cp <= 0x1F) || cp == 0x85 ||
        cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000
    end

    def is_cjk(cp : Int32) : Bool
      (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) ||
        (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) ||
        (cp >= 0x2B820 && cp <= 0x2CEAF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x2F800 && cp <= 0x2FA1F)
    end

    # Python's SentencePiece normalizer drops Cc, preserving Cf and joiners.
    def is_control(cp : Int32) : Bool
      (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) && !is_space(cp)
    end

    private def ascii_punct(cp : Int32) : Bool
      (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)
    end

    # ── normalization ───────────────────────────────────────────────────
    # Lowercase a string using the full Unicode mapping, applying the
    # contextual final-sigma rule.
    def downcase(text : String) : String
      cps = text.each_codepoint.to_a
      lowered = String.build(text.bytesize) do |io|
        before_cased = false
        i = 0
        while i < cps.size
          cp = cps[i]
          # Final sigma: a capital sigma preceded by a cased character and
          # followed by nothing cased (ignoring case-ignorable characters)
          # lowercases to the final form.
          final_sigma = false
          if cp == 0x3A3 && before_cased
            j = i + 1
            nxt = 0
            while j < cps.size
              nxt = cps[j]
              break unless is_case_ignorable(nxt)
              j += 1
            end
            final_sigma = j == cps.size || !is_cased(nxt)
          end
          before_cased = is_cased(cp) unless is_case_ignorable(cp)
          low = lower_cp(cp)
          low[0] = 0x3C2 if final_sigma
          low.each { |c| io << c.chr }
          i += 1
        end
      end
      lowered
    end

    # Canonical decomposition (NFD) then removal of combining marks — the
    # second half of BERT's "strip accents" step.
    def strip_marks(text : String) : String
      decomposed = text.unicode_normalize(:nfd)
      String.build(decomposed.bytesize) do |io|
        decomposed.each_codepoint do |cp|
          io << cp.chr unless is_mn(cp)
        end
      end
    end

    # BERT-style normalization: lowercase, decompose, strip marks, and put
    # spaces around CJK characters.
    def bert_normalize(text : String) : String
      lowered = downcase(text)
      stripped = strip_marks(lowered)
      String.build(stripped.bytesize * 2) do |io|
        last_space = true
        stripped.each_codepoint do |cp|
          if is_space(cp)
            unless last_space
              io << ' '
              last_space = true
            end
            next
          end
          if is_cjk(cp)
            io << ' ' unless last_space
            io << cp.chr
            io << ' '
            last_space = true
            next
          end
          io << cp.chr
          last_space = false
        end
      end
    end

    # SentencePiece normalization: NFKC, drop control characters, collapse
    # whitespace, and isolate punctuation.
    def sp_normalize(text : String) : String
      buf = String.build(text.bytesize * 2) do |io|
        last_space = true
        text.unicode_normalize(:nfkc).each_codepoint do |cp|
          next if is_control(cp)
          if is_space(cp)
            unless last_space
              io << ' '
              last_space = true
            end
            next
          end
          # SentencePiece isolates ASCII punctuation only — not all Unicode
          # punctuation.  This matches the reference implementation exactly.
          if ascii_punct(cp)
            unless last_space
              io << ' '
            end
            io << cp.chr
            io << ' '
            last_space = true
            next
          end
          io << cp.chr
          last_space = false
        end
      end
      buf.strip
    end
  end
end
