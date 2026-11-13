# maxsat_cli.cr — command line front end for the Crystal NuWLS core.
#
#   crystal run src/maxsat_cli.cr -- instance.wcnf --time-budget-seconds 30
#
# This intentionally mirrors SUTRA's useful flags without linking or
# reusing the C solver.
require "json"
require "./maxsat"

module Semrember
  module MaxSat
    module Cli
      extend self

      def usage(io : IO)
        io.puts "Usage: maxsat <instance.cnf|instance.wcnf> [options]"
        io.puts "  --time-budget-seconds X   wall-clock budget (0 = unlimited)"
        io.puts "  --restarts N              restart budget (default 4)"
        io.puts "  --steps N                 continuous steps per restart (default 300)"
        io.puts "  --repair-flips N          discrete flips per restart (default 1000000)"
        io.puts "  --seed N                  deterministic seed (default 42)"
        io.puts "  --solution FILE           write the witness as DIMACS v-lines"
        io.puts "  --warm-start FILE         seed the continuous phase from a witness"
        io.puts "  --no-json                 human-readable output"
        io.puts "  --help                    this message"
      end

      def load_warm(path : String, nv : Int32) : Array(UInt8)
        warm = Array(UInt8).new(nv + 1, 0_u8)
        File.read_lines(path).each do |raw|
          line = raw.strip
          next if line.empty? || line.starts_with?('c') || line.starts_with?('p')
          line = line[1..] if line.starts_with?('v') || line.starts_with?('V')
          line.split.each do |tok|
            lit = tok.to_i?
            next unless lit
            v = lit.abs
            warm[v] = lit > 0 ? 1_u8 : 0_u8 if v >= 1 && v <= nv
          end
        end
        warm
      end

      def write_solution(path : String, asn : Array(UInt8)) : Nil
        File.open(path, "w") do |f|
          f.puts "s SATISFIABLE"
          0.upto(asn.size - 1) do |i|
            v = i + 1
            f.puts "v #{asn[i] == 1 ? v : -v}"
          end
        end
      end

      def run(argv : Array(String)) : Int32
        if argv.empty? || {"-h", "--help", "help"}.includes?(argv[0])
          usage(STDOUT)
          return 0
        end

        input = nil.as(String?)
        opts = Options.new
        solution = nil.as(String?)
        warm_path = nil.as(String?)

        i = 0
        while i < argv.size
          arg = argv[i]
          case arg
          when "--time-budget-seconds"
            raise ArgumentError.new("missing value for #{arg}") if i + 1 >= argv.size
            opts.time_budget = argv[i + 1].to_f
            i += 2
          when "--restarts"
            opts.restarts = argv[i + 1].to_i
            i += 2
          when "--steps"
            opts.steps = argv[i + 1].to_i
            i += 2
          when "--repair-flips"
            opts.repair_flips = argv[i + 1].to_i64
            i += 2
          when "--seed"
            opts.seed = argv[i + 1].to_u64
            i += 2
          when "--solution"
            solution = argv[i + 1]
            i += 2
          when "--warm-start"
            warm_path = argv[i + 1]
            i += 2
          when "--no-json"
            opts.json = false
            i += 1
          when "--json"
            opts.json = true
            i += 1
          else
            raise ArgumentError.new("unknown option #{arg}") if arg.starts_with?('-')
            input = arg
            i += 1
          end
        end

        raise ArgumentError.new("missing input file") unless input
        f = Wcnf.parse(File.read(input.not_nil!))
        opts.warm_start = load_warm(warm_path, f.nv) if warm_path
        result = Solver.solve(f, opts)

        if opts.json
          payload = JSON.build do |j|
            j.object do
              j.field "engine", "MaxSatCr"
              j.field "version", VERSION
              j.field "input", input.not_nil!
              j.field "status", result.status
              j.field "variables", f.nv
              j.field "clauses", f.nc
              j.field "hard_unsatisfied", result.hard_unsatisfied
              j.field "soft_cost", result.soft_cost
              j.field "flips", result.flips
              j.field "restarts", result.restarts
              j.field "converged", result.converged
              j.field "wall_seconds", result.wall
              j.field "verified", "recount_from_witness"
            end
          end
          puts payload
        else
          puts "MaxSatCr #{VERSION}  #{input}: #{result.status}"
          puts "  hard unsat #{result.hard_unsatisfied}, soft cost #{result.soft_cost}"
          puts "  flips=#{result.flips} restarts=#{result.restarts} " \
               "converged=#{result.converged} wall=#{result.wall.round(3)}s"
        end

        write_solution(solution.not_nil!, result.assignment) if solution
        0
      end
    end
  end
end

begin
  exit Semrember::MaxSat::Cli.run(ARGV)
rescue ex
  STDERR.puts "error: #{ex.message}"
  exit 1
end
