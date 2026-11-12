require "spec"
require "../src/semrember"
require "../src/csv"

describe "F16" do
  it "round-trips exactly representable values" do
    Semrember::F16.to_f32(0x3C00_u16).should eq 1.0_f32
    Semrember::F16.to_f32(0x3000_u16).should eq 0.125_f32
    Semrember::F16.to_f32(0x3020_u16).should eq 0.12890625_f32
    Semrember::F16.to_f32(0x0000_u16).should eq 0.0_f32
  end

  it "keeps the mantissa" do
    # A mantissa of 32 shifted into float32's 23-bit field.  Getting this
    # wrong truncates every weight to its power-of-two precision.
    Semrember::F16.to_f32(0x3020_u16).should_not eq 0.125_f32
  end

  it "round-trips through the encoder" do
    [0.12890625_f32, 1.0_f32, -0.00828552_f32, 0.0_f32].each do |v|
      Semrember::F16.to_f32(Semrember::F16.from_f32(v)).should be_close(v, 1e-7)
    end
  end
end

describe "Codec" do
  it "round-trips a negative float32" do
    # Negative values have the sign bit set, so a UInt32 round trip through
    # Int32 overflows.  Reinterpreting must not.
    [-3.4635426998138428_f32, -0.5_f32, -1e-9_f32, 0.0_f32, 1.0_f32].each do |v|
      b = Semrember::Codec.f32(v)
      Semrember::Codec.read_f32(b, 0).should eq v
    end
  end

  it "round-trips a negative int32" do
    [-1, Int32::MIN, 0, 1, Int32::MAX].each do |v|
      b = Semrember::Codec.u32(v)
      Semrember::Codec.read_u32(b, 0).should eq v.unsafe_as(UInt32)
    end
  end

  it "is little-endian" do
    Semrember::Codec.u32(1).should eq Bytes[1, 0, 0, 0]
    Semrember::Codec.u16(1).should eq Bytes[1, 0]
  end
end

describe "Linalg.invert" do
  it "inverts a well-conditioned matrix" do
    m = [4.0, 1.0, 2.0, 0.5,
         1.0, 3.0, 0.2, 1.0,
         2.0, 0.2, 5.0, 0.3,
         0.5, 1.0, 0.3, 2.0].flat_map { |r| r }
    inv = Semrember::Linalg.invert(m, 4)
    # A * A⁻¹ = I
    4.times do |i|
      4.times do |j|
        s = 0.0
        4.times { |k| s += m[i * 4 + k] * inv[k * 4 + j] }
        s.should be_close(i == j ? 1.0 : 0.0, 1e-9)
      end
    end
  end

  it "pivots below the diagonal" do
    # A zero in the leading position forces a row swap; without the pivot
    # search the inverse is garbage.
    m = [0.0, 2.0,
         1.0, 3.0].flat_map { |r| r }
    inv = Semrember::Linalg.invert(m, 2)
    2.times do |i|
      2.times do |j|
        s = 0.0
        2.times { |k| s += m[i * 2 + k] * inv[k * 2 + j] }
        s.should be_close(i == j ? 1.0 : 0.0, 1e-12)
      end
    end
  end

  it "matches Gauss-Jordan on an SPD system via Cholesky" do
    m = [4.0, 1.0, 2.0, 0.5,
         1.0, 3.0, 0.2, 1.0,
         2.0, 0.2, 5.0, 0.3,
         0.5, 1.0, 0.3, 2.0].flat_map { |r| r }
    expected = Semrember::Linalg.invert(m, 4)
    actual = Semrember::Linalg.cholesky_inverse(m, 4)
    16.times { |i| actual[i].should be_close(expected[i], 1e-10) }
  end

  it "multiplies row-major matrices" do
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # 2x3
    b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0] # 3x2
    c = Semrember::Linalg.matmul(a, b, 2, 3, 2)
    c.should eq [58.0, 64.0, 139.0, 154.0]
  end

  it "deletes a principal submatrix exactly" do
    m = [4.0, 1.0, 2.0, 0.5,
         1.0, 3.0, 0.2, 1.0,
         2.0, 0.2, 5.0, 0.3,
         0.5, 1.0, 0.3, 2.0].flat_map { |r| r }
    a = Semrember::Linalg.invert(m, 4)
    # Delete row/column 1, then invert the retained 3x3 block.
    retained = [0, 2, 3].flat_map do |r|
      [0, 2, 3].map { |c| m[r * 4 + c] }
    end
    expected = Semrember::Linalg.invert(retained, 3)
    actual = Semrember::Linalg.delete_principal(a, 4, 1)
    9.times { |i| actual[i].should be_close(expected[i], 1e-10) }
  end
end

describe "Csv" do
  it "parses quoted fields, embedded commas and CRLF" do
    rows = Semrember::Csv.parse_csv("a,b\r\n\"x,y\",\"z\"\r\n\"say \"\"hi\"\"\",q\r\n")
    rows[0].should eq ["a", "b"]
    rows[1].should eq ["x,y", "z"]
    rows[2].should eq ["say \"hi\"", "q"]
  end

  it "keeps bytes verbatim" do
    rows = Semrember::Csv.parse_csv("k,v\nna\xc3\xafve,x\n")
    rows[1][0].bytesize.should eq 6
  end
end

describe "AssociativeMemory" do
  corpus = {
    "No space left on device" => "free_disk",
    "write failed: ENOSPC" => "free_disk",
    "could not resolve host api.example.com" => "fix_dns",
    "certificate has expired" => "renew_certificate",
  }

  it "classifies what it was trained on" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    mem.query("No space left on device", 1)[0][0].should eq "free_disk"
    mem.query("certificate has expired", 1)[0][0].should eq "renew_certificate"
  end

  it "lower-cases keys when building the vocabulary" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    # "ENOSPC" must not create a separate uppercase term.
    mem.d_tfidf.should be > 0
    mem.encode("WRITE FAILED: ENOSPC").idx.size.should be > 0
  end

  it "saves and loads an AMEM file" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    path = File.tempname("semrember", ".amem")
    begin
      mem.save(path)
      loaded = Semrember::Memory.load(path)
      loaded.labels.should eq mem.labels
      loaded.d_tfidf.should eq mem.d_tfidf
      loaded.query("No space left on device", 1)[0][0].should eq "free_disk"
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "replaces a key exactly on the dual writer" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    mem.set("No space left on device", "clear_cache")
    mem.query("No space left on device", 1)[0][0].should eq "clear_cache"
    # The neighbouring key is untouched.
    mem.query("write failed: ENOSPC", 1)[0][0].should eq "free_disk"
  end

  it "deletes a key" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    mem.delete("No space left on device").should be_true
    mem.delete("No space left on device").should be_false
  end

  it "reports novelty in range, low for known input" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    mem.novelty("No space left on device").should be < 0.2
    mem.novelty("zqxwvbnmwk").should be > 0.5
  end

  it "finds similar stored keys" do
    mem = Semrember::Memory.new(solver: :dual).fit(corpus)
    mem.similar("write failed ENOSPC", 1)[0][0].should eq "write failed: ENOSPC"
  end
end

describe "Unicode" do
  it "lowercases with the contextual final-sigma rule" do
    # Python's str.lower() maps the second sigma to the final form; a
    # codepoint-wise mapping does not.
    Semrember::Unicode.downcase("ΣΣ").should eq "σς"
    Semrember::Unicode.downcase("ΟΣ").should eq "ος"
  end

  it "maps capital I with a dot to i with a dot" do
    Semrember::Unicode.downcase("İ").should eq "i̇"
  end

  it "strips combining marks" do
    Semrember::Unicode.strip_marks("é").should eq "e"
  end
end
