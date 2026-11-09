require "spec"
require "http/client"
require "../src/server"

def with_kemal(&block : HTTP::Handler, Semrember::Server ->)
  config = Kemal.config
  config.clear
  routes = Kemal::RouteHandler::INSTANCE
  routes.routes = Radix::Tree(Kemal::Route).new
  routes.cached_routes = Kemal::LRUCache(String, Radix::Result(Kemal::Route)).new(config.max_route_cache_size)
  service = Semrember::Server.new(nil, nil)
  service.configure_kemal("127.0.0.1", 7379)
  config.logging = false
  config.setup
  handler = HTTP::Server.build_middleware(config.handlers)
  yield handler, service
ensure
  config.try(&.clear)
  if router = routes
    router.routes = Radix::Tree(Kemal::Route).new
    router.cached_routes = Kemal::LRUCache(String, Radix::Result(Kemal::Route)).new(1024)
  end
end

def kemal_request(handler : HTTP::Handler, method : String, path : String, body : String? = nil)
  output = IO::Memory.new
  req = HTTP::Request.new(method, path, body: body)
  res = HTTP::Server::Response.new(output)
  handler.call(HTTP::Server::Context.new(req, res))
  res.close
  output.rewind
  HTTP::Client::Response.from_io(output, ignore_body: method == "HEAD")
end

describe "Kemal HTTP integration" do
  it "removes a wave of expired keys from exact and semantic reads" do
    with_kemal do |handler, service|
      ["disk", "host", "cert"].each do |key|
        kemal_request(handler, "PUT", "/v1/kv/#{key}", {value: key}.to_json).status_code.should eq 200
      end
      expired = Time.utc - 1.second
      service.expiry["disk"] = expired
      service.expiry["host"] = expired
      service.expiry["cert"] = Time.utc + 1.hour

      response = kemal_request(handler, "POST", "/v1/similar", {text: "disk", top_k: 3}.to_json)
      JSON.parse(response.body)["candidates"].as_a.map { |candidate| candidate["key"].as_s }.should eq ["cert"]
      kemal_request(handler, "GET", "/v1/kv/disk").status_code.should eq 404
      kemal_request(handler, "GET", "/v1/kv/host").status_code.should eq 404
      kemal_request(handler, "GET", "/v1/kv/cert").status_code.should eq 200
      service.expiry.keys.should eq ["cert"]
    end
  end

  it "mounts the versioned routes without changing JSON responses" do
    with_kemal do |handler, _service|
      ping = kemal_request(handler, "GET", "/v1/ping")
      ping.status_code.should eq 200
      JSON.parse(ping.body)["ok"].as_bool.should be_true
      kemal_request(handler, "PUT", "/v1/kv/disk", {value: "x"}.to_json).status_code.should eq 200
      hit = kemal_request(handler, "POST", "/v1/similar", {text: "disk"}.to_json)
      JSON.parse(hit.body)["candidates"][0]["value"].as_s.should eq "x"
      Kemal.config.serve_static.should be_false
      Kemal.config.host_binding.should eq "127.0.0.1"
    end
  end

  it "preserves domain JSON errors while rendering framework errors as JSON" do
    with_kemal do |handler, _service|
      [
        {"GET", "/v1/kv/missing", nil, 404, "not found"},
        {"PUT", "/v1/kv/disk", "{}", 400, "value must be a string"},
        {"GET", "/missing", nil, 404, "Not Found"},
        {"PUT", "/v1/ping", nil, 405, "Method Not Allowed"},
      ].each do |method, path, body, status, error|
        response = kemal_request(handler, method, path, body)
        response.status_code.should eq status
        response.content_type.should eq "application/json"
        JSON.parse(response.body)["error"].as_s.should eq error
      end
    end
  end

  it "preserves encoded slashes, literal plus, Unicode and empty keys" do
    with_kemal do |handler, _service|
      ["a/exists", "a+b", "東京", ""].each do |key|
        path = "/v1/kv/" + URI.encode_path_segment(key)
        kemal_request(handler, "PUT", path, {value: key}.to_json).status_code.should eq 200
        response = kemal_request(handler, "GET", path)
        response.status_code.should eq 200
        JSON.parse(response.body)["value"].as_s.should eq key
      end
    end
  end

  it "handles HEAD through GET with no response body" do
    with_kemal do |handler, _service|
      head = kemal_request(handler, "HEAD", "/v1/ping")
      head.status_code.should eq 200
      head.body.should be_empty
    end
  end

  it "does not expose static files or redirect encoded keys" do
    with_kemal do |handler, _service|
      kemal_request(handler, "GET", "/README.md").status_code.should eq 404
      path = "/v1/kv/a%2F%2Fb"
      kemal_request(handler, "PUT", path, {value: "x"}.to_json).status_code.should eq 200
      response = kemal_request(handler, "GET", path)
      response.status_code.should eq 200
      response.headers.has_key?("Location").should be_false
    end
  end
end
