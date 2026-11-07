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
end
