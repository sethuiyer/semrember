# csv.cr — RFC 4180 CSV reader.
#
# Quoted fields, embedded commas and newlines, doubled quotes, CRLF, and a
# leading BOM.  The reference implementations parse the same dialect, so the
# two agree on the awkward rows.
#
# Fields are accumulated as raw bytes rather than characters: a CSV field is
# a byte string, and decoding it as characters first would corrupt any
# non-UTF-8 payload.

module Semrember
  module Csv
    extend self

    def parse_csv(text : String) : Array(Array(String))
      bytes = text.to_slice
      off = 0
      # Skip a UTF-8 BOM.
      if bytes.size >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF
        off = 3
      end

      rows = [] of Array(String)
      row = [] of String
      field = IO::Memory.new
      in_quotes = false
      field_started = false

      while off < bytes.size
        c = bytes[off]
        if in_quotes
          if c == '"'.ord
            if off + 1 < bytes.size && bytes[off + 1] == '"'.ord
              field.write_byte('"'.ord.to_u8)
              off += 1
            else
              in_quotes = false
            end
          else
            field.write_byte(c)
          end
        else
          if c == '"'.ord && !field_started && field.empty?
            in_quotes = true
            field_started = true
          elsif c == ','.ord
            row << field.to_s
            field = IO::Memory.new
            field_started = false
          elsif c == '\n'.ord
            f = field.to_s
            f = f.byte_slice(0, f.bytesize - 1) if f.ends_with?('\r')
            row << f unless f.empty? && row.empty?
            rows << row
            row = [] of String
            field = IO::Memory.new
            field_started = false
          else
            field.write_byte(c)
            field_started = true
          end
        end
        off += 1
      end
      row << field.to_s unless field.empty?
      rows << row unless row.empty?
      rows
    end

    # Load a two-column CSV into ordered (key, value) pairs.  Order matters:
    # the label order in a fitted model follows the order of the rows, so the
    # reference implementation and this one agree on it.
    def load_pairs(path : String, key_col : String, val_col : String) : Array(Tuple(String, String))
      rows = parse_csv(File.read(path))
      raise "#{path}: no data rows" if rows.size < 2
      header = rows[0]
      kc = header.index(key_col)
      vc = header.index(val_col)
      raise "#{path}: column not found (key=#{key_col} val=#{val_col})" if kc.nil? || vc.nil?
      pairs = [] of Tuple(String, String)
      rows[1..].each do |r|
        next if r.size <= kc.not_nil! || r.size <= vc.not_nil!
        k = r[kc.not_nil!].strip
        v = r[vc.not_nil!].strip
        next if k.empty?
        pairs << {k, v}
      end
      pairs
    end
  end
end
