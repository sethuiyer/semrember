require "./linalg"

# Standalone read-side index. No Memory/HTTP integration or persistence.
module Semrember::Mips
  {% if flag?(:linux) %}
    @[Link("m")]
    lib LibEnvironment
      fun fegetround : Int32
    end
  {% end %}

  # The interval primitives assume round-to-nearest and gradual underflow.
  # Linux FE_TONEAREST is zero. Other targets conservatively request a scan
  # until their floating-point environment has an explicit implementation.
  def self.supported_environment?(value : Float32) : Bool
    {% if flag?(:linux) %}
      return false unless LibEnvironment.fegetround == 0
      # Runtime-dependent signs prevent compile-time folding of these probes.
      sign32 = value.unsafe_as(UInt32) & 0x80000000_u32
      small32 = (sign32 | 0x00800000_u32).unsafe_as(Float32)
      half32 = small32 * 0.5_f32
      return false unless half32.to_f64 == small32.to_f64 * 0.5 && half32 * 2_f32 == small32
      sign64 = sign32.to_u64 << 32
      small64 = (sign64 | 0x0010000000000000_u64).unsafe_as(Float64)
      half64 = small64 * 0.5
      half64 != 0.0 && half64 * 2.0 == small64
    {% else %}
      false
    {% end %}
  end

  # IEEE binary64 round-to-nearest operations widened by one representable
  # value. No process-global rounding-mode changes. Never use fast-math.
  module Outward
    extend self

    def up(x : Float64) : Float64
      return x if x == Float64::INFINITY || x.nan?
      return 1_u64.unsafe_as(Float64) if x == 0.0
      bits = x.unsafe_as(UInt64)
      (x > 0 ? bits + 1_u64 : bits - 1_u64).unsafe_as(Float64)
    end

    def down(x : Float64) : Float64
      -up(-x)
    end

    def add(a : Float64, b : Float64) : Float64
      up(a + b)
    end

    def mul(a : Float64, b : Float64) : Float64
      up(a * b)
    end

    # Upper norm of intervals, including construction and sqrt rounding.
    def norm(values : Array(Interval)) : Float64
      sum = 0.0
      values.each do |v|
        a = Math.max(v.lo.abs, v.hi.abs)
        sum = add(sum, mul(a, a))
      end
      up(Math.sqrt(sum))
    end
  end

  struct Interval
    getter lo : Float64
    getter hi : Float64

    def initialize(@lo : Float64, @hi : Float64)
    end

    def self.point(x : Float64) : self
      new(x, x)
    end

    def +(b : self) : self
      self.class.new(Outward.down(@lo + b.lo), Outward.up(@hi + b.hi))
    end

    def -(b : self) : self
      self.class.new(Outward.down(@lo - b.hi), Outward.up(@hi - b.lo))
    end

    def *(b : self) : self
      p = {@lo*b.lo, @lo*b.hi, @hi*b.lo, @hi*b.hi}
      self.class.new(Outward.down(p.min), Outward.up(p.max))
    end
  end

  record Config, rank : Int32 = 32, leaf_size : Int32 = 16,
    seed : Int32 = 0, power_iters : Int32 = 2,
    max_nodes : Int32 = Int32::MAX, max_labels : Int32 = Int32::MAX

  record Hit, label : Int32, score : Float32
  record SearchResult, hits : Array(Hit), nodes_visited : Int32 = 0,
    labels_scored : Int32 = 0, pruned_nodes : Int32 = 0, fallback : Bool = false do
    def fallback? : Bool
      fallback
    end
  end

  # Right randomized subspace iteration, followed by a small Rayleigh-Ritz
  # eigensystem. The stored basis need not be orthonormal for correctness.
  class Basis
    getter columns : Array(Array(Float64))

    def initialize(w : Array(Float32), t : Int32, d : Int32, config : Config,
                   supplied : Array(Array(Float64))? = nil)
      if supplied
        raise ArgumentError.new("basis shape/nonfinite") unless supplied.all? { |v| v.size == d && v.all?(&.finite?) }
        @columns = supplied.map(&.dup)
        return
      end
      rank = Math.min(config.rank, Math.min(t, d))
      width = Math.min(rank + 8, Math.min(t, d))
      if rank == 0
        @columns = [] of Array(Float64)
        return
      end
      rng = Random.new(config.seed.to_u64)
      columns = Array.new(width) do
        omega = Array.new(t) { rng.rand * 2.0 - 1.0 }
        Array.new(d) { |j| (0...t).sum { |i| w[i*d + j].to_f64 * omega[i] } }
      end
      columns = Linalg.orthonormalize(columns)
      config.power_iters.times do
        columns = columns.map do |v|
          y = Array.new(t) { |i| (0...d).sum { |j| w[i*d + j].to_f64 * v[j] } }
          Array.new(d) { |j| (0...t).sum { |i| w[i*d + j].to_f64 * y[i] } }
        end
        columns = Linalg.orthonormalize(columns)
      end
      width = columns.size
      if width == 0
        @columns = [] of Array(Float64)
        return
      end
      h = Array.new(t) { |i| columns.map { |v| (0...d).sum { |j| w[i*d + j].to_f64 * v[j] } } }
      gram = Array.new(width * width) { |ij| (0...t).sum { |i| h[i][ij // width] * h[i][ij % width] } }
      _, rotation = Linalg.jacobi_eigen(gram, width)
      @columns = Array.new(Math.min(rank, width)) do |k|
        Array.new(d) { |j| (0...width).sum { |a| columns[a][j] * rotation[a*width + k] } }
      end
      raise ArgumentError.new("nonfinite basis") unless @columns.all? { |v| v.all?(&.finite?) }
    end
  end

  private record Node, center : Array(Float64), radius : Float64,
    residual : Float64, weight_norm : Float64, first : Int32, count : Int32,
    left : Int32 = -1, right : Int32 = -1

  # Best-bound-first queue. Lower node id wins bound ties.
  class Heap
    @items = [] of Tuple(Float64, Int32)

    def empty? : Bool
      @items.empty?
    end

    private def better(a, b) : Bool
      a[0] > b[0] || (a[0] == b[0] && a[1] < b[1])
    end

    def push(bound : Float64, id : Int32)
      item = {bound, id}
      @items << item
      i = @items.size - 1
      while i > 0
        p = (i - 1) // 2
        break unless better(item, @items[p])
        @items[i] = @items[p]
        i = p
      end
      @items[i] = item
    end

    def pop : Tuple(Float64, Int32)
      result = @items[0]
      last = @items.pop
      unless @items.empty?
        i = 0
        while (child = i*2 + 1) < @items.size
          child += 1 if child + 1 < @items.size && better(@items[child + 1], @items[child])
          break unless better(@items[child], last)
          @items[i] = @items[child]
          i = child
        end
        @items[i] = last
      end
      result
    end
  end

  class Index
    getter version : Int64
    getter label_count : Int32
    getter dimension : Int32
    @config : Config
    @columns : Array(Array(Float64))
    @h : Array(Array(Float64))
    @residuals : Array(Float64)
    @norms : Array(Float64)
    @order : Array(Int32)
    @nodes = [] of Node

    def initialize(w : Array(Float32), @label_count : Int32, @dimension : Int32,
                   @config = Config.new, @version = 0_i64,
                   basis : Array(Array(Float64))? = nil)
      raise ArgumentError.new("unsupported floating-point environment") unless Mips.supported_environment?(1_f32)
      raise ArgumentError.new("weight shape") unless @label_count >= 0 && @dimension >= 0 && w.size.to_i64 == @label_count.to_i64 * @dimension
      raise ArgumentError.new("index config") unless @config.rank >= 0 && @config.leaf_size > 0 && @config.power_iters >= 0 && @config.seed >= 0 && @config.max_nodes >= 0 && @config.max_labels >= 0
      raise ArgumentError.new("nonfinite weights") unless w.all?(&.finite?)
      @columns = Basis.new(w, @label_count, @dimension, @config, basis).columns
      @order = (0...@label_count).to_a
      @h = Array.new(@label_count) do |i|
        @columns.map { |v| (0...@dimension).sum { |j| v[j] * w[i*@dimension + j].to_f64 } }
      end
      @norms = Array.new(@label_count) do |i|
        Outward.norm(Array.new(@dimension) { |j| Interval.point(w[i*@dimension + j].to_f64) })
      end
      # h is a stored Float64 vector. Certify the actual residual w - Vh,
      # not an idealized result of the projection algorithm.
      @residuals = Array.new(@label_count) do |i|
        Outward.norm(Array.new(@dimension) do |j|
          sum = Interval.point(0.0)
          @columns.each_with_index { |v, k| sum += Interval.point(v[j]) * Interval.point(@h[i][k]) }
          Interval.point(w[i*@dimension + j].to_f64) - sum
        end)
      end
      raise ArgumentError.new("unbounded construction") unless @h.all? { |row| row.all?(&.finite?) } && @norms.all?(&.finite?) && @residuals.all?(&.finite?)
      build(0, @label_count) if @label_count > 0
    end

    def rank : Int32
      @columns.size
    end

    def node_count : Int32
      @nodes.size
    end

    # Numeric payload only; excludes Crystal object/allocator overhead.
    def payload_bytes : Int64
      8_i64 * (rank.to_i64 * @dimension + @label_count.to_i64 * rank + 2_i64 * @label_count + @nodes.size.to_i64 * (rank + 3)) + 4_i64 * (@label_count + 4_i64 * @nodes.size)
    end

    private def key(id : Int32, axis : Int32)
      {@h[id][axis], id}
    end

    # Deterministic quickselect with a bounded number of partitions; sort
    # fallback prevents quadratic behavior. Worst-case tree split work is
    # O(T log^2 T), usual split work O(T log T).
    private def nth(first : Int32, count : Int32, mid : Int32, axis : Int32)
      lo, hi = first, first + count - 1
      budget = 2 * Math.log2(count.to_f64 + 1).to_i + 2
      while lo < hi
        if budget <= 0
          sorted = @order[lo..hi].sort_by { |id| key(id, axis) }
          sorted.each_with_index { |id, k| @order[lo + k] = id }
          return
        end
        budget -= 1
        p = (lo + hi) // 2
        pivot = key(@order[p], axis)
        @order[p], @order[hi] = @order[hi], @order[p]
        store = lo
        (lo...hi).each do |j|
          if key(@order[j], axis) < pivot
            @order[j], @order[store] = @order[store], @order[j]
            store += 1
          end
        end
        @order[store], @order[hi] = @order[hi], @order[store]
        return if store == mid
        if store < mid
          lo = store + 1
        else
          hi = store - 1
        end
      end
    end

    private def build(first : Int32, count : Int32) : Int32
      ids = @order[first, count]
      center = Array.new(rank) { |k| ids.sum { |id| @h[id][k] / count } }
      radius = ids.max_of do |id|
        Outward.norm(Array.new(rank) { |k| Interval.point(@h[id][k]) - Interval.point(center[k]) })
      end
      node = Node.new(center, radius, ids.max_of { |i| @residuals[i] }, ids.max_of { |i| @norms[i] }, first, count)
      raise ArgumentError.new("unbounded node") unless radius.finite? && center.all?(&.finite?)
      index = @nodes.size
      @nodes << node
      if count > @config.leaf_size
        if rank > 0
          axis = (0...rank).max_by { |k| ids.sum { |id| (@h[id][k] - center[k]) ** 2 } }
          nth(first, count, first + count // 2, axis)
        end
        left = build(first, count // 2)
        right = build(first + count // 2, count - count // 2)
        @nodes[index] = Node.new(center, radius, node.residual, node.weight_norm, first, count, left, right)
      end
      index
    end

    private def projection(x : Array(Float32)) : Array(Interval)
      @columns.map do |v|
        sum = Interval.point(0.0)
        @dimension.times { |j| sum += Interval.point(v[j]) * Interval.point(x[j].to_f64) }
        sum
      end
    end

    private def bound(node : Node, u : Array(Interval), unorm : Float64,
                      xnorm : Float64, gamma : Float64, tiny : Float64) : Float64
      dot = Interval.point(0.0)
      rank.times { |k| dot += Interval.point(node.center[k]) * u[k] }
      real = Outward.add(dot.hi, Outward.mul(node.radius, unorm))
      real = Outward.add(real, Outward.mul(node.residual, xnorm))
      rounding = Outward.mul(gamma, Outward.mul(node.weight_norm, xnorm))
      Outward.add(Outward.add(real, rounding), tiny)
    end

    private def error_terms(n : Int32) : Tuple(Float64, Float64)
      # Standard relative + absolute-error recurrence for at most 2n
      # Float32 operations. Extra factor 1/(1-2nu) also covers propagation
      # of gradual-underflow errors. Reject nonpositive denominators.
      ku = n.to_f64 * 2.0 * (2.0 ** -24)
      return {Float64::INFINITY, Float64::INFINITY} if ku >= 1.0
      denom = Outward.down(1.0 - ku)
      gamma = Outward.up(ku / denom)
      tiny = Outward.up((n.to_f64 * 2.0 * (2.0 ** -149)) / denom)
      {gamma, tiny}
    end

    # Diagnostics return copies, never mutable internal nodes/basis.
    def bounds(x : Array(Float32), terms : Int32 = @dimension) : Array(Tuple(Array(Int32), Float64))
      raise ArgumentError.new("query shape/terms") unless x.size == @dimension && terms >= @dimension && x.all?(&.finite?)
      u = projection(x)
      xn = Outward.norm(x.map { |v| Interval.point(v.to_f64) })
      un = Outward.norm(u)
      gamma, tiny = error_terms(terms)
      @nodes.map { |node| {@order[node.first, node.count], bound(node, u, un, xn, gamma, tiny)} }
    end

    # The caller must score the SAME immutable W snapshot and feature values,
    # using <= terms multiplications and additions, without fast-math/FTZ/DAZ.
    # No approximate hits are returned when fallback is requested.
    def search(x : Array(Float32), topk : Int32, expected_version : Int64 = @version,
               terms : Int32 = @dimension, &score : Array(Int32) -> Array(Float32)) : SearchResult
      raise ArgumentError.new("query shape/topk/terms") unless x.size == @dimension && topk >= 0 && terms >= @dimension
      fail = SearchResult.new([] of Hit, fallback: true)
      return fail unless expected_version == @version && x.all?(&.finite?)
      return SearchResult.new([] of Hit) if topk == 0 || @label_count == 0
      return fail if topk >= @label_count || @dimension == 0 || x.all? { |v| v == 0 }
      return fail unless Mips.supported_environment?(x[0])
      u = projection(x)
      xn = Outward.norm(x.map { |v| Interval.point(v.to_f64) })
      un = Outward.norm(u)
      gamma, tiny = error_terms(terms)
      # Bound absolute intermediate sums to exclude Float32 overflow, even
      # if the true final score would be finite after cancellation.
      magnitude = Outward.mul(@nodes[0].weight_norm, xn)
      limit = Outward.add(Outward.mul(Outward.add(1.0, gamma), magnitude), tiny)
      return fail unless limit.finite? && limit < Float32::MAX.to_f64 && un.finite?
      heap = Heap.new
      root = bound(@nodes[0], u, un, xn, gamma, tiny)
      return fail unless root.finite?
      heap.push(root, 0)
      hits = [] of Hit
      visited = scored = pruned = 0
      until heap.empty?
        upper, id = heap.pop
        if hits.size == topk && upper < hits.last.score.to_f64
          pruned += 1
          next
        end
        return SearchResult.new([] of Hit, visited, scored, pruned, true) if visited >= @config.max_nodes
        visited += 1
        node = @nodes[id]
        if node.left < 0
          return SearchResult.new([] of Hit, visited, scored, pruned, true) if node.count > @config.max_labels - scored
          labels = @order[node.first, node.count]
          values = yield labels.dup
          raise ArgumentError.new("scorer result shape") unless values.size == labels.size
          return SearchResult.new([] of Hit, visited, scored, pruned, true) unless values.all?(&.finite?)
          scored += labels.size
          labels.each_with_index do |label, i|
            hits << Hit.new(label, values[i])
          end
          hits.sort_by! { |hit| {-hit.score, hit.label} }
          hits = hits.first(topk)
        else
          {node.left, node.right}.each do |child|
            upper = bound(@nodes[child], u, un, xn, gamma, tiny)
            return SearchResult.new([] of Hit, visited, scored, pruned, true) unless upper.finite?
            if hits.size == topk && upper < hits.last.score.to_f64
              pruned += 1
            else
              heap.push(upper, child)
            end
          end
        end
      end
      SearchResult.new(hits, visited, scored, pruned)
    end
  end
end
