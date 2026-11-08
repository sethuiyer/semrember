require "spec"
require "http/client"
require "../src/server"

def request(server : Semrember::Server, method : String, target : String, body : String? = nil)
  output = IO::Memory.new
  req = HTTP::Request.new(method, target, body: body)
  res = HTTP::Server::Response.new(output)
  server.handle(HTTP::Server::Context.new(req, res))
  res.close
  output.rewind
  parsed = HTTP::Client::Response.from_io(output)
  {parsed.status_code, JSON.parse(parsed.body)}
end

def put(server : Semrember::Server, key : String, value : String)
  request(server, "PUT", "/v1/kv/" + URI.encode_path_segment(key), {value: value}.to_json)
end

describe Semrember::Server do
  it "learns writes into an empty service and returns previous exact values" do
    server = Semrember::Server.new(nil, nil)
    status, body = put(server, "disk full", "free_disk")
    status.should eq 200
    body["previous"].raw.should be_nil
    body["semantic_updated"].as_bool.should be_true
    _, body = put(server, "disk full", "clear_cache")
    body["previous"].as_s.should eq "free_disk"
    _, body = request(server, "POST", "/v1/similar", {text: "disk full"}.to_json)
    body["candidates"][0]["value"].as_s.should eq "clear_cache"
    _, body = request(server, "POST", "/v1/escalate", {text: "disk full"}.to_json)
    body["action"].as_s.should eq "clear_cache"
  end

  it "calculates margins from the two highest-ranked labels for the whole corpus" do
    server = Semrember::Server.new(nil, nil, File.join(__DIR__, "../examples/incidents.csv"))
    Semrember::Csv.load_pairs(File.join(__DIR__, "../examples/incidents.csv"), "message", "action").each do |key, label|
      status, body = request(server, "POST", "/v1/escalate", {text: key}.to_json)
      status.should eq 200
      candidates = body["candidates"].as_a
      first = candidates[0]["score"].as_f
      second = candidates[1]["score"].as_f
      expected = (first - second) / (first.abs + second.abs)
      body["margin"].as_f.should be_close(expected, 1e-12)
      body["handled"].as_bool.should be_true
      body["action"].as_s.should eq label
    end
  end

  it "purges expired records from EXISTS, TTL, semantic reads and info" do
    ["exists", "ttl", "similar", "escalate", "novelty", "info"].each do |operation|
      server = Semrember::Server.new(nil, nil)
      put(server, "disk", "old_action")
      server.expiry["disk"] = Time.utc - 1.second
      status, body = case operation
                     when "exists", "ttl"
                       request(server, "GET", "/v1/kv/disk/#{operation}")
                     when "similar", "escalate"
                       request(server, "POST", "/v1/#{operation}", {text: "disk"}.to_json)
                     else
                       request(server, "GET", "/v1/#{operation}?text=disk")
                     end
      status.should eq 200
      case operation
      when "exists"   then body["exists"].as_bool.should be_false
      when "ttl"      then body["ttl"].raw.should be_nil
      when "similar"  then body["candidates"].as_a.should be_empty
      when "escalate" then body["handled"].as_bool.should be_false
      when "novelty"  then body["novelty"].as_f.should eq 1.0
      when "info"     then body["records"].as_i.should eq 0
      end
      request(server, "GET", "/v1/kv/disk")[0].should eq 404
      put(server, "host", "new_action")[0].should eq 200
    end
  end

  it "does not revive expired records when EXPIRE is called again" do
    server = Semrember::Server.new(nil, nil)
    put(server, "disk", "x")
    server.expiry["disk"] = Time.utc - 1.second
    _, body = request(server, "POST", "/v1/kv/disk/expire", {seconds: 60}.to_json)
    body["ok"].as_bool.should be_false
  end

  it "requires a complete glob match and handles empty suffixes and Unicode" do
    server = Semrember::Server.new(nil, nil)
    ["review", "review_suffix", "é", ""].each { |key| put(server, key, "x") }
    _, body = request(server, "GET", "/v1/kv?match=review")
    body["keys"].as_a.map(&.as_s).should eq ["review"]
    _, body = request(server, "GET", "/v1/kv?match=?")
    body["keys"].as_a.map(&.as_s).should eq ["é"]
    _, body = request(server, "GET", "/v1/kv?match=*x")
    body["keys"].as_a.map(&.as_s).should eq ["review_suffix"]
    _, body = request(server, "GET", "/v1/kv?match=*")
    body["keys"].as_a.size.should eq 4
  end

  it "round-trips reserved suffixes, encoded slashes, plus and Unicode keys" do
    server = Semrember::Server.new(nil, nil)
    ["a/exists", "b/ttl", "c/expire", "a+b", "東京", "a%2Fexists"].each do |key|
      put(server, key, "value")[0].should eq 200
      path = "/v1/kv/" + URI.encode_path_segment(key)
      status, body = request(server, "GET", path)
      status.should eq 200
      body["value"].as_s.should eq "value"
      _, body = request(server, "GET", path + "/exists")
      body["exists"].as_bool.should be_true
      request(server, "POST", path + "/expire", {seconds: 60}.to_json)[0].should eq 200
      _, body = request(server, "GET", path + "/ttl")
      body["ttl"].as_i.should be >= 0
    end
  end

  it "keeps differently cased records in the semantic index after delete" do
    server = Semrember::Server.new(nil, nil)
    put(server, "Disk", "upper")
    put(server, "disk", "lower")
    request(server, "DELETE", "/v1/kv/Disk")[0].should eq 200
    _, body = request(server, "POST", "/v1/similar", {text: "disk"}.to_json)
    body["candidates"].as_a.map { |c| c["key"].as_s }.should eq ["disk"]
  end

  it "validates JSON before mutating existing records" do
    server = Semrember::Server.new(nil, nil)
    put(server, "disk", "keep")
    ["{", "[]", "{}", "{\"value\": 12}"].each do |body|
      request(server, "PUT", "/v1/kv/disk", body)[0].should eq 400
      request(server, "GET", "/v1/kv/disk")[1]["value"].as_s.should eq "keep"
    end
    request(server, "POST", "/v1/similar", {text: "disk", top_k: -1}.to_json)[0].should eq 400
    request(server, "GET", "/v1/kv?cursor=-1")[0].should eq 400
    request(server, "GET", "/v1/kv?count=0")[0].should eq 400
    request(server, "GET", "/v1/kv?cursor=999")[1]["keys"].as_a.should be_empty
    request(server, "GET", "/v1/kv?cursor=2147483647")[1]["keys"].as_a.should be_empty
    request(server, "PUT", "/v1/kv/%FF", {value: "invalid"}.to_json)[0].should eq 400
    request(server, "POST", "/v1/kv")[0].should eq 405
  end

  it "reports unavailable semantic state explicitly for loaded models" do
    path = File.tempname("semrember-server", ".amem")
    begin
      Semrember::Memory.new.fit({"disk" => "x"}).save(path)
      server = Semrember::Server.new(path, nil)
      _, body = put(server, "disk", "y")
      body["semantic_updated"].as_bool.should be_false
      request(server, "GET", "/v1/kv/disk")[1]["value"].as_s.should eq "y"
      request(server, "POST", "/v1/similar", {text: "disk"}.to_json)[0].should eq 409
      request(server, "POST", "/v1/escalate", {text: "disk"}.to_json)[0].should eq 409
      expect_raises(ArgumentError) { Semrember::Server.new(path, nil, "unused.csv") }
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
