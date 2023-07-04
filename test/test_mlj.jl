import SymbolicRegression: SRRegressor, MultitargetSRRegressor
import MLJTestInterface as MTI
import MLJBase: machine, fit!, report
using Test

@testset "Generic interface tests" begin
    failures, summary = MTI.test(
        [SRRegressor], MTI.make_regression()...; mod=@__MODULE__, verbosity=0, throw=true
    )
    @test isempty(failures)

    X = randn(100, 3)
    Y = @. cos(X^2) * 3.2 - 0.5
    (X, Y) = MTI.table.((X, Y))
    failures, summary = MTI.test(
        [MultitargetSRRegressor], X, Y; mod=@__MODULE__, verbosity=0, throw=true
    )
    @test isempty(failures)
end

@testset "Variable names" begin
    @testset "Single outputs" begin
        X = (a=rand(32), b=rand(32))
        y = X.a .^ 2.1
        model = SRRegressor(; niterations=10)
        mach = machine(model, X, y)
        fit!(mach)
        rep = report(mach)
        @test occursin("a", rep.equation_strings[rep.best_idx])
    end

    @testset "Multiple outputs" begin
        X = (a=rand(32), b=rand(32))
        y = X.a .^ 2.1
        model = MultitargetSRRegressor(; niterations=10)
        mach = machine(model, X, reduce(hcat, [reshape(y, :, 1) for i in 1:3]))
        fit!(mach)
        rep = report(mach)
        @test all(
            eq -> occursin("a", eq), [rep.equation_strings[i][rep.best_idx[i]] for i in 1:3]
        )
    end
end
