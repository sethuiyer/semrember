# codec.cr — little-endian primitives.
#
# Written out explicitly rather than via IO::ByteFormat so the byte order is
# a property of the file formats, not of the host.
#
# The encoders reinterpret bits instead of converting them.  `Int32#to_u32`
# and `UInt32#to_i32` *raise* on values that do not fit, so a negative
# float32 (anything with the sign bit set) would overflow through a
# conversion — silently in `--release`, which disables overflow checks.
# Reinterpreting cannot overflow, and is what these functions mean.

module Semrember
  module Codec
    extend self

    def u32(v : Int32) : Bytes
      x = v.unsafe_as(UInt32)
      b = Bytes.new(4)
      4.times { |i| b[i] = ((x >> (8 * i)) & 0xFF).to_u8 }
      b
    end

    def u16(v : Int32) : Bytes
      x = v.unsafe_as(UInt32)
      b = Bytes.new(2)
      2.times { |i| b[i] = ((x >> (8 * i)) & 0xFF).to_u8 }
      b
    end

    def f32(v : Float32) : Bytes
      u32(v.unsafe_as(Int32))
    end

    def read_u32(b : Bytes, off : Int32) : UInt32
      b[off].to_u32 | (b[off + 1].to_u32 << 8) |
        (b[off + 2].to_u32 << 16) | (b[off + 3].to_u32 << 24)
    end

    def read_u16(b : Bytes, off : Int32) : UInt16
      (b[off].to_u32 | (b[off + 1].to_u32 << 8)).to_u16
    end

    def read_f32(b : Bytes, off : Int32) : Float32
      read_u32(b, off).unsafe_as(Float32)
    end
  end

  # ── float16 (IEEE 754 binary16), round-to-nearest-even ────────────────
  # AMEM stores weights as float16.  numpy and the C library agree bit for
  # bit on this conversion, so files are byte-identical across implementations.
  module F16
    extend self

    def to_f32(h : UInt16) : Float32
      sign = (h & 0x8000_u16).to_u32 << 16
      exp = (h >> 10).to_u32 & 0x1F_u32
      # The mask must be UInt16: mixing widths leaves the result UInt16, and
      # `mant << 13` would then overflow and wrap to zero, silently
      # truncating every weight to its power-of-two precision.
      mant = (h & 0x03FF_u16).to_u32
      bits =
        if exp == 0
          if mant == 0
            sign
          else
            e = -14
            m = mant
            while (m & 0x0400_u32) == 0
              m <<= 1
              e -= 1
            end
            m &= 0x03FF_u32
            sign | ((e + 127).to_u32 << 23) | (m << 13)
          end
        elsif exp == 31
          sign | 0x7F80_0000_u32 | (mant << 13)
        else
          sign | ((exp + 112_u32) << 23) | (mant << 13)
        end
      bits.unsafe_as(Float32)
    end

    def from_f32(f : Float32) : UInt16
      x = f.unsafe_as(UInt32)
      sign = ((x >> 16) & 0x8000_u32).to_u16
      absx = x & 0x7FFF_FFFF_u32
      if absx >= 0x7F80_0000_u32
        return absx > 0x7F80_0000_u32 ? (sign | 0x7E00_u16) : (sign | 0x7C00_u16)
      end
      exp = (absx >> 23).to_i32 & 0xFF
      mant = absx & 0x007F_FFFF_u32
      return sign if exp == 0
      e = exp - 127 + 15
      return sign | 0x7C00_u16 if e >= 31
      if e <= 0
        return sign if e < -10
        mant |= 0x0080_0000_u32
        shift = 14 - e
        half = mant >> shift
        rem = mant & ((1_u32 << shift) - 1)
        halfway = 1_u32 << (shift - 1)
        half += 1 if rem > halfway || (rem == halfway && (half & 1) == 1)
        return sign | half.to_u16
      end
      half = (e.to_u32 << 10) | (mant >> 13)
      rem = mant & 0x1FFF_u32
      half += 1 if rem > 0x1000_u32 || (rem == 0x1000_u32 && (half & 1) == 1)
      return sign | 0x7C00_u16 if half >= 0x7C00_u32
      sign | half.to_u16
    end
  end
end
