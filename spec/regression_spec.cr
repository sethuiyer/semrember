require "spec"
require "../src/semrember"
require "../src/csv"

def with_test_encoder(dim : Int32 = 2, &block : String ->)
  path = File.tempname("semrember-encoder", ".bin")
  begin
    File.open(path, "wb") do |f|
      f << "M2VE"
      [1, 3, dim].each { |v| f.write(Semrember::Codec.u32(v)) }
      ["[UNK]", "disk", "host"].each do |token|
        f.write(Semrember::Codec.u16(token.bytesize))
        f << token
      end
      3.times do |i|
        dim.times { |d| f.write(Semrember::Codec.u16(Semrember::F16.from_f32(d == i % dim ? 1_f32 : 0_f32))) }
      end
    end
    yield path
  ensure
    File.delete(path) if File.exists?(path)
  end
end

describe "tokenizer regressions" do
  it "matches base-8M WordPiece IDs for ASCII symbols and Greek sigma" do
    # IDs from the shipped tokenizer.json, with only the tokens used here.
    tok = Semrember::WordPiece.new({
      "[UNK]" => 1, "+" => 15, "=" => 33, "a" => 43, "b" => 44, "c" => 45,
      "ο" => 175, "key" => 2151, "value" => 2649, "##σ" => 28739,
      "process" => 1838, "crashed" => 7013,
    })
    tok.encode("key=value").should eq [2151, 33, 2649]
    tok.encode("a+b=c").should eq [43, 15, 44, 33, 45]
    tok.encode("C++ process crashed").should eq [45, 15, 15, 1838, 7013]
    tok.encode("ΟΣ").should eq [175, 28739]
  end

  it "drops BERT control, format and replacement characters before splitting" do
    tok = Semrember::WordPiece.new({"[UNK]" => 1, "disk" => 8791, "full" => 1446})
    ["dis\u0000k", "dis\u0001k", "dis\u001ck", "\uFEFFdisk", "dis\uFFFDk",
     "dis\u200Dk", "dis\u00ADk", "dis\uE000k"].each do |input|
      tok.encode(input).should eq [8791]
    end
    tok.encode("disk\tfull\ndisk\rfull").should eq [8791, 1446, 8791, 1446]
  end

  it "matches multilingual IDs for control, format and replacement characters" do
    tok = Semrember::UnigramSP.new({
      "<unk>" => 1, "▁dis" => 2835, "▁disk" => 28336, "▁k" => 470, "k" => 90,
      "\u00AD" => 229,
    })
    ["\uFEFFdisk", "dis\u0001k", "dis\u001ck"].each do |input|
      tok.encode(input).should eq [28336]
    end
    ["dis\uFFFDk", "dis\u200Dk", "dis\uFEFFk"].each do |input|
      tok.encode(input).should eq [2835, 470]
    end
    tok.encode("dis\u00ADk").should eq [2835, 229, 90]
    tok.encode("dis\u0000k").should eq [2835, 1, 90]
  end

  it "excludes WordPiece unknowns from the mean while exposing the raw IDs" do
    with_test_encoder do |path|
      encoder = Semrember::StaticEncoder.new(path)
      encoder.encode_ids("disk 🙂").should eq [1, 0]
      encoder.encode("disk 🙂").should eq encoder.encode("disk")
      encoder.encode("🙂").should be_empty
    end
  end

  it "keeps the lowercase table unchanged across calls and contexts" do
    3.times do
      Semrember::Unicode.downcase("ΣΣ").should eq "σς"
      Semrember::Unicode.downcase("ΣΑ").should eq "σα"
      Semrember::Unicode.downcase("ΟΣ\u0301").should eq "ος\u0301"
      Semrember::Unicode::LOWER[0x3A3_u32].should eq [0x3C3_u32]
    end
  end

  it "isolates CJK exactly once in original input order" do
    tok = Semrember::WordPiece.new({"[UNK]" => 0, "中" => 1, "a" => 2, "文" => 3})
    tok.encode("a中文a").should eq [2, 1, 3, 2]
  end
end

describe "CSV boundary regressions" do
  it "preserves trailing empty fields with and without a newline" do
    ["x,", "x,\n", "x,\r\n"].each do |input|
      Semrember::Csv.parse_csv(input).should eq [["x", ""]]
    end
    Semrember::Csv.parse_csv("\"\"").should eq [[""]]
  end

  it "preserves quoted CR and newlines verbatim" do
    Semrember::Csv.parse_csv("a,\"x\r\"\r\nb,\"y\r\nz\"\r\n").should eq [["a", "x\r"], ["b", "y\r\nz"]]
  end

  it "rejects unterminated quotes" do
    expect_raises(ArgumentError) { Semrember::Csv.parse_csv("x,\"open") }
  end
end

describe "memory update regressions" do
  it "returns each similar key once without discarding training weights" do
    mem = Semrember::Memory.new.fit({"disk full" => "free_disk", "host down" => "fix_dns"},
      {"disk full" => 3})
    mem.stored.should eq 4
    mem.similar("disk full", 2).map(&.[0]).should eq ["disk full", "host down"]
  end

  it "deletes a batch of keys including weighted rows and prunes unused labels" do
    mem = Semrember::Memory.new.fit({"disk full" => "x", "host down" => "y", "bad cert" => "z"},
      {"disk full" => 3})
    mem.novelty("disk full")
    mem.delete_many(["disk full", "host down", "disk full", "missing"]).should be_true
    mem.delete_many(["missing"]).should be_false
    mem.stored.should eq 1
    mem.labels.should eq ["z"]
    mem.similar("disk full").map(&.[0]).should eq ["bad cert"]
    mem.novelty("disk full").should be > 0.9
    mem.query("bad cert")[0][0].should eq "z"
  end

  it "bootstraps a fresh or explicitly empty dual writer" do
    [Semrember::Memory.new, Semrember::Memory.new.fit({} of String => String)].each do |mem|
      mem.set("disk full", "free_disk").should be_nil
      mem.query("disk full")[0][0].should eq "free_disk"
      mem.similar("disk full")[0][0].should eq "disk full"
      mem.novelty("disk full").should be < 0.1
    end
  end

  it "returns the old target on overwrite and nil on insert" do
    mem = Semrember::Memory.new.fit({"disk full" => "free_disk"})
    mem.set("disk full", "clear_cache").should eq "free_disk"
    mem.set("full disk", "free_disk").should be_nil
    mem.stored.should eq 2
    mem.delete("full disk").should be_true
    mem.stored.should eq 1
  end

  it "keeps key identity case-sensitive while features remain lowercase" do
    mem = Semrember::Memory.new.fit({"Disk" => "upper", "disk" => "lower"})
    mem.set("Disk", "changed").should eq "upper"
    mem.set("disk", "lower2").should eq "lower"
    mem.delete("Disk").should be_true
    mem.similar("disk").map(&.[0]).should eq ["disk"]
    mem.query("disk")[0][0].should eq "lower2"
  end

  it "accepts another write after deletion of the last observation" do
    mem = Semrember::Memory.new.fit({"one" => "x"})
    mem.delete("one").should be_true
    mem.delete("one").should be_false
    mem.similar("one").should be_empty
    mem.novelty("one").should eq 1.0
    mem.set("two", "y").should be_nil
    mem.query("two")[0][0].should eq "y"
  end

  it "matches a full ridge refit after repeated target replacements" do
    corpus = {"disk full" => "x", "host down" => "y", "bad cert" => "x"}
    mem = Semrember::Memory.new.fit(corpus)
    ["z", "y", "x", "z"].each do |value|
      mem.set("disk full", value)
      corpus["disk full"] = value
      reference = Semrember::Memory.new.fit(corpus)
      ["disk full", "host down", "cert bad", "unknown"].each do |query|
        actual = mem.labels.zip(mem.scores(query)).to_h
        expected = reference.labels.zip(reference.scores(query)).to_h
        actual.keys.sort.should eq expected.keys.sort
        expected.each { |label, score| actual[label].should be_close(score, 1e-6) }
      end
    end
  end

  it "handles zero-dimensional live observations without division by zero" do
    mem = Semrember::Memory.new(min_df: 100).fit({"a" => "x"})
    mem.set("b", "y").should be_nil
    mem.set("b", "z").should eq "y"
    mem.delete("a").should be_true
    mem.stored.should eq 1
  end

  it "invalidates novelty on observation changes and preserves it on relabeling" do
    mem = Semrember::Memory.new.fit({"disk full" => "x", "host down" => "y"})
    old = mem.novelty("full disk")
    mem.novelty("full disk").should eq old
    mem.set("disk full", "z")
    mem.novelty("full disk").should eq old
    mem.set("full disk", "x")
    inserted = mem.novelty("full disk")
    inserted.should be < old
    mem.delete("full disk")
    mem.novelty("full disk").should be_close(old, 1e-10)
    mem.fit({"new corpus" => "n"})
    mem.novelty("new corpus").should be < 0.1
  end

  it "rejects negative limits and invalid regularization" do
    mem = Semrember::Memory.new
    expect_raises(ArgumentError) { mem.query("x", -1) }
    expect_raises(ArgumentError) { mem.similar("x", -1) }
    expect_raises(ArgumentError) { Semrember::Memory.new(rls_reg: Float64::NAN) }
  end

  it "reports singular inverses instead of returning a zero matrix" do
    expect_raises(ArgumentError, "singular matrix") { Semrember::Linalg.invert([1.0, 1.0, 1.0, 1.0], 2) }
  end
end

describe "AMEM loading regressions" do
  it "requires a matching encoder for hybrid models" do
    path = File.tempname("semrember-hybrid", ".amem")
    begin
      with_test_encoder do |encoder_path|
        encoder = Semrember::StaticEncoder.new(encoder_path)
        mem = Semrember::Memory.new(static: encoder).fit({"disk" => "x", "host" => "y"})
        mem.save(path)
        expect_raises(ArgumentError, "requires static encoder dimension 2, got 0") { Semrember::Memory.load(path) }
        loaded = Semrember::Memory.load(path, static: encoder)
        loaded.query("disk")[0][0].should eq "x"
        loaded.scores("disk").zip(mem.scores("disk")).each { |a, b| a.should be_close(b, 0.001) }
        with_test_encoder(3) do |wrong|
          expect_raises(ArgumentError, "requires static encoder dimension 2, got 3") do
            Semrember::Memory.load(path, static: Semrember::StaticEncoder.new(wrong))
          end
        end
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "distinguishes loaded readers from empty writable memories" do
    path = File.tempname("semrember-reader", ".amem")
    begin
      Semrember::Memory.new.fit({"disk" => "x"}).save(path)
      mem = Semrember::Memory.load(path)
      mem.writable?.should be_false
      expect_raises(Semrember::UnavailableError) { mem.set("disk", "y") }
      expect_raises(Semrember::UnavailableError) { mem.similar("disk") }
      mem.fit({"host" => "y"})
      mem.writable?.should be_true
      mem.set("host", "z").should eq "y"
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects truncated headers and payloads" do
    path = File.tempname("semrember-truncated", ".amem")
    begin
      File.write(path, "AM")
      expect_raises(ArgumentError) { Semrember::Memory.load(path) }
      Semrember::Memory.new.fit({"disk" => "x"}).save(path)
      data = File.read(path)
      File.write(path, data.byte_slice(0, data.bytesize - 1))
      expect_raises(ArgumentError) { Semrember::Memory.load(path) }
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
