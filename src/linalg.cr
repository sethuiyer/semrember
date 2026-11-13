{% if flag?(:lapack) %}
  @[Link("lapack")]
  @[Link("blas")]
  lib LibSemremberLapack
    fun dpotrf_(uplo : UInt8*, n : Int32*, a : Float64*, lda : Int32*, info : Int32*) : Void
    fun dpotri_(uplo : UInt8*, n : Int32*, a : Float64*, lda : Int32*, info : Int32*) : Void
    fun dgemm_(transa : UInt8*, transb : UInt8*, m : Int32*, n : Int32*, k : Int32*,
               alpha : Float64*, a : Float64*, lda : Int32*, b : Float64*, ldb : Int32*,
               beta : Float64*, c : Float64*, ldc : Int32*) : Void
  end
{% end %}

module Semrember
  # ── dense linear algebra ─────────────────────────────────────────────
  module Linalg
    extend self

    # Gauss-Jordan inversion with partial pivoting, row-major.
    def invert(m : Array(Float64), n : Int32) : Array(Float64)
      a = m.dup
      inv = Array(Float64).new(n * n, 0_f64)
      n.times { |i| inv[i * n + i] = 1.0 }
      n.times do |col|
        piv = col
        best = a[col * n + col].abs
        (col + 1...n).each do |r|
          v = a[r * n + col].abs
          if v > best
            best = v
            piv = r
          end
        end
        raise ArgumentError.new("singular matrix") if best < 1e-12
        if piv != col
          n.times { |c| a[col * n + c], a[piv * n + c] = a[piv * n + c], a[col * n + c] }
          n.times { |c| inv[col * n + c], inv[piv * n + c] = inv[piv * n + c], inv[col * n + c] }
        end
        d = a[col * n + col]
        n.times { |c| a[col * n + c] /= d; inv[col * n + c] /= d }
        n.times do |r|
          next if r == col
          f = a[r * n + col]
          next if f == 0.0
          n.times { |c| a[r * n + c] -= f * a[col * n + c]; inv[r * n + c] -= f * inv[col * n + c] }
        end
      end
      inv
    end

    # Cholesky factorization of a symmetric positive-definite matrix,
    # row-major lower-triangular L such that M = L L^T.
    def cholesky(m : Array(Float64), n : Int32) : Array(Float64)
      {% if flag?(:lapack) %}
        # LAPACK uses column-major storage.  K is symmetric, so the lower
        # triangle in row-major is the same matrix; copy the result back into
        # a row-major lower factor.
        work = m.dup
        uplo = 'L'.ord.to_u8
        nn = n
        lda = n
        info = 0
        LibSemremberLapack.dpotrf_(pointerof(uplo), pointerof(nn), work.to_unsafe, pointerof(lda), pointerof(info))
        raise ArgumentError.new("matrix is not positive definite") unless info == 0
        l = Array(Float64).new(n * n, 0_f64)
        n.times do |i|
          (0..i).each { |j| l[i * n + j] = work[i + j * n] }
        end
        l
      {% else %}
        l = Array(Float64).new(n * n, 0_f64)
        n.times do |i|
          (i...n).each do |j|
            s = m[i * n + j]
            i.times { |k| s -= l[i * n + k] * l[j * n + k] }
            if i == j
              raise ArgumentError.new("matrix is not positive definite") if s <= 0.0
              l[i * n + i] = Math.sqrt(s)
            else
              l[j * n + i] = s / l[i * n + i]
            end
          end
        end
        l
      {% end %}
    end

    # Inverse of an SPD matrix from its Cholesky factor.  Solving for the
    # inverse costs N^3/3 + N^3/3, roughly half of Gauss-Jordan's work.
    def cholesky_inverse(m : Array(Float64), n : Int32) : Array(Float64)
      {% if flag?(:lapack) %}
        work = m.dup
        uplo = 'L'.ord.to_u8
        nn = n
        lda = n
        info = 0
        LibSemremberLapack.dpotrf_(pointerof(uplo), pointerof(nn), work.to_unsafe, pointerof(lda), pointerof(info))
        raise ArgumentError.new("matrix is not positive definite") unless info == 0
        LibSemremberLapack.dpotri_(pointerof(uplo), pointerof(nn), work.to_unsafe, pointerof(lda), pointerof(info))
        raise ArgumentError.new("matrix is singular") unless info == 0
        inv = Array(Float64).new(n * n, 0_f64)
        n.times do |i|
          (0..i).each do |j|
            v = work[i + j * n]
            inv[i * n + j] = v
            inv[j * n + i] = v
          end
        end
        inv
      {% else %}
        l = cholesky(m, n)
        inv_l = Array(Float64).new(n * n, 0_f64)
        n.times do |col|
          n.times do |i|
            s = (i == col ? 1.0 : 0.0)
            i.times { |k| s -= l[i * n + k] * inv_l[k * n + col] }
            inv_l[i * n + col] = s / l[i * n + i]
          end
        end
        inv = Array(Float64).new(n * n, 0_f64)
        n.times do |i|
          (i...n).each do |j|
            s = 0.0
            n.times { |k| s += inv_l[k * n + i] * inv_l[k * n + j] }
            inv[i * n + j] = s
            inv[j * n + i] = s
          end
        end
        inv
      {% end %}
    end

    # Cholesky first, Gauss-Jordan as a numerical fallback.
    def inverse_spd(m : Array(Float64), n : Int32) : Array(Float64)
      cholesky_inverse(m, n)
    rescue ArgumentError
      invert(m, n)
    end

    # C = A * B for row-major A (m×k) and B (k×n).  With LAPACK we pass the
    # row-major operands as column-major transposes; the resulting C^T in
    # column-major order is exactly C in row-major order.
    def matmul(a : Array(Float64), b : Array(Float64), m : Int32, k : Int32, n : Int32) : Array(Float64)
      {% if flag?(:lapack) %}
        c = Array(Float64).new(m * n, 0_f64)
        trans = 'N'.ord.to_u8
        mm = m
        nn = n
        kk = k
        alpha = 1.0
        beta = 0.0
        lda = n
        ldb = k
        ldc = n
        # Fortran: C^T(n×m) = B^T(n×k) * A^T(k×m)
        LibSemremberLapack.dgemm_(pointerof(trans), pointerof(trans), pointerof(nn), pointerof(mm),
          pointerof(kk), pointerof(alpha), b.to_unsafe, pointerof(lda), a.to_unsafe,
          pointerof(ldb), pointerof(beta), c.to_unsafe, pointerof(ldc))
        c
      {% else %}
        c = Array(Float64).new(m * n, 0_f64)
        m.times do |i|
          n.times { |j| c[i * n + j] = 0.0 }
          k.times do |p|
            aip = a[i * k + p]
            next if aip == 0.0
            n.times { |j| c[i * n + j] += aip * b[p * n + j] }
          end
        end
        c
      {% end %}
    end

    # Inverse of the principal submatrix obtained by deleting row/column i.
    # A' = A_RR - A_Ri A_iR / A_ii.  One outer product, O(N^2).
    def delete_principal(a : Array(Float64), n : Int32, i : Int32) : Array(Float64)
      n1 = n - 1
      out = Array(Float64).new(n1 * n1, 0_f64)
      aii = a[i * n + i]
      raise ArgumentError.new("singular principal submatrix") if aii.abs < 1e-12
      ri = 0
      n.times do |r|
        next if r == i
        ci = 0
        n.times do |c|
          next if c == i
          out[ri * n1 + ci] = a[r * n + c] - a[r * n + i] * a[i * n + c] / aii
          ci += 1
        end
        ri += 1
      end
      out
    end
  end
end

module Semrember::Linalg
  def transpose(a : Array(Float64), rows : Int32, cols : Int32) : Array(Float64)
    raise ArgumentError.new("matrix shape") unless a.size == rows * cols
    Array(Float64).new(a.size) { |i| a[(i % rows) * cols + i // rows] }
  end

  # Two-pass modified Gram-Schmidt. Dependent columns are omitted.
  def orthonormalize(columns : Array(Array(Float64))) : Array(Array(Float64))
    result = [] of Array(Float64)
    columns.each do |column|
      v = column.dup
      2.times do
        result.each do |q|
          dot = v.zip(q).sum { |a, b| a * b }
          v.size.times { |i| v[i] -= dot * q[i] }
        end
      end
      norm = Math.sqrt(v.sum { |x| x * x })
      next unless norm.finite? && norm > 1e-12
      result << v.map { |x| x / norm }
    end
    result
  end

  # Symmetric Jacobi eigensystem; vectors are columns, descending eigenvalues.
  # Used only to choose a search basis, never to certify a pruning bound.
  def jacobi_eigen(m : Array(Float64), n : Int32) : Tuple(Array(Float64), Array(Float64))
    raise ArgumentError.new("matrix shape") unless n >= 0 && m.size == n * n
    a = m.dup
    v = Array(Float64).new(n * n, 0.0)
    n.times { |i| v[i * n + i] = 1.0 }
    50.times do
      changed = false
      n.times do |p|
        (p + 1...n).each do |q|
          apq = a[p * n + q]
          next if apq.abs <= 1e-14 * (1.0 + a[p * n + p].abs + a[q * n + q].abs)
          changed = true
          theta = 0.5 * Math.atan2(2.0 * apq, a[q * n + q] - a[p * n + p])
          c = Math.cos(theta)
          s = Math.sin(theta)
          app = a[p * n + p]
          aqq = a[q * n + q]
          n.times do |i|
            next if i == p || i == q
            aip, aiq = a[i * n + p], a[i * n + q]
            a[i * n + p] = a[p * n + i] = c * aip - s * aiq
            a[i * n + q] = a[q * n + i] = s * aip + c * aiq
          end
          a[p * n + p] = c*c*app - 2*c*s*apq + s*s*aqq
          a[q * n + q] = s*s*app + 2*c*s*apq + c*c*aqq
          a[p * n + q] = a[q * n + p] = 0.0
          n.times do |i|
            vip, viq = v[i * n + p], v[i * n + q]
            v[i * n + p] = c * vip - s * viq
            v[i * n + q] = s * vip + c * viq
          end
        end
      end
      break unless changed
    end
    order = (0...n).to_a.sort_by { |i| {-a[i * n + i], i} }
    {order.map { |i| a[i * n + i] }, Array(Float64).new(n*n) { |i| v[(i // n)*n + order[i % n]] }}
  end
end
