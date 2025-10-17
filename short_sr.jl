using SymbolicRegression
using SymbolicUtils

function discover_short_formula(X::Matrix{Float64}, y::Vector{Float64}, feature_names::Vector{String}, max_complexity::Int=10, n_iterations::Int=40)
    options = Options(
        binary_operators=[+, *, /, -],
        unary_operators=[cos, exp, sin, log, sqrt],
        populations=20,
        maxsize=max_complexity,  # Enforce shorter formulas by limiting tree size
        model_selection=:score   # Penalize complexity for concise, accurate formulas
    )

    hall_of_fame = equation_search(
        X, y,
        niterations=n_iterations,
        options=options,
        parallelism=:multithreading
    )

    dominating = calculate_pareto_frontier(hall_of_fame)
    best_member = dominating[end]  # Select the highest-scoring (accurate + short)

    # Convert to SymPy-compatible string and replace variables (e.g., x1 -> Feature1)
    tree = best_member.tree
    equation_str = string(SymbolicUtils.simplify(tree))
    for (i, name) in enumerate(feature_names)
        equation_str = replace(equation_str, "x$(i)" => name)
    end

    score = best_member.score
    complexity = best_member.complexity

    return equation_str, score, complexity
end
