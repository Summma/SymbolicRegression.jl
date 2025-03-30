using Pkg
Pkg.activate(".")
Pkg.instantiate()

push!(LOAD_PATH, "src/")

using SymbolicRegression

X = randn(Float32, 5, 100)
y = 1 ./ (2 * cos.(X[4, :])) + X[1, :] .^ 2 .- 2

inv(x) = 1 / x

options = SymbolicRegression.Options(;
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp, inv],
    # bin_op_weight=[0.33, 0.33, 0.01, 0.33],
    # un_op_weight=[0.1, 0, 0.9],
)

hall_of_fame = equation_search(X, y; options=options, parallelism=:multithreading)

dominating = calculate_pareto_frontier(hall_of_fame)

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
