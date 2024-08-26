# <42137 Optimization using Metaheuristics -- Assignment 05>
# ALNS for PlastOut Production Planning
#
# This script encapsulates the implementation of an Adaptive Large Neighborhood Search (ALNS) 
# for addressing the production planning problem at PlastOut A/S. The primary objective is to 
# maximize the revenue by effectively planning the assignment of orders to production lines within a 
# specified time horizon, accounting for individual product revenues and additional revenue from 
# pairing products on the same production line.
#
# The ALNS employs the following strategies and components:
#   Initialization: Generates a feasible initial solution that respects production line capacities 
#                   and maximizes initial revenue using a deterministic heuristic.
#   Destroy Operators: Two methods are implemented to remove orders from the production lines,
#                      promoting solution space exploration:
#                      1. Apocalypse - Randomly removes a subset of orders.
#                      2. Ragnarok - Targets specific orders and their neighbors for removal.
#   Repair Operator: A greedy heuristic, Bob_The_Builder, that reinserts orders to optimize the current solution, 
#                    aiming for revenue increment.
#   Adaptive Mechanism: Dynamically adjusts the probability of employing destroy operators based 
#                       on their past performance, balancing diversification and intensification.
#                       Reheating is applied for the simulated annealing after an extended period of iterations.
#   Acceptance Criterion: A combination of Improving Solution and a Simulated Annealing acceptance strategy is utilized to escape local optima 
#                         by accepting non-improving moves with a certain probability.
#   Termination Criteria: The algorithm concludes when a predefined computational time has been reached,
#                         ensuring efficient execution.
#   Objective Calculation: Revenue calculation considers both individual order revenues and pairwise 
#                          revenue boosts for co-located orders.
#*****************************************************************************************************
using Random

function read_instance(filename)
    f = open(filename)
    name = readline(f) # name of the instance
    size = parse(Int32,readline(f)) # number of order
    LB = parse(Int32,readline(f)) # best known revenue
    rev = parse.(Int32,split(readline(f)))# revenue for including an order
    rev_pair = zeros(Int32,size,size) # pairwise revenues
    for i in 1:size-1
        data = parse.(Int32,split(readline(f)))
        j=i+1
        for d in data
            rev_pair[i,j]=d
            rev_pair[j,i]=d
            j+=1
        end
    end
    readline(f)
    k = parse(Int32,readline(f)) # number of production lines
    H = parse(Int32,readline(f)) # planning horizon
    p = parse.(Int32,split(readline(f))) # production times
    close(f)
    return name, size, LB ,rev, rev_pair, k, H, p
end
#*****************************************************************************************************


#*****************************************************************************************************
# Definition of ProductionPlanningSolver
# A struct that encapsulates the essential elements for addressing the Production Planning Problem within a Adaptive Large Neighborhood Search framework. 
# It includes information about individual product revenues, pairwise product cost savings when produced on the same line, production times, 
# the number of production lines (k), the planning horizon (H), and the total number of products (size).
struct ProductionPlanningSolver
    rev::Array{Int32,1}
    rev_pair::Array{Int32,2}
    p::Array{Int32,1}
    k::Int32
    H::Int32
    size::Int32
    # The constructor initializes these elements
    ProductionPlanningSolver(rev, rev_pair, p, k, H, size) = new(rev, rev_pair, p, k, H, size)
end

# Definition of ProductionPlanningSolution
# A mutable struct that symbolizes a feasible solution for the Production Planning Problem. 
# It maintains an assignment of products to production lines, the total objective value of the solution, 
# and the production times utilized on each line. This structure facilitates the evaluation of solution quality 
# and compliance with production constraints.
mutable struct ProductionPlanningSolution
    # we represent a solution as an array of arrays, each sub-array represents a production line
    assignment::Array{Array{Int32,1},1}
    # placeholder for the objective value
    objective::Int32
    # total production time for each line
    production_times::Array{Int32,1}
    # remaininig products to add
    remaining_products::Array{Int32,1}

    # Default constructor for creating an empty solution setup
    ProductionPlanningSolution(k, H) = new([Int32[] for _ in 1:k], 0, zeros(Int32, k) .+ H, Int32[])  # starts with empty remaining_products
end

function elapsedTime(startTime)
    return round((time_ns()-startTime)/1e9, digits=3)
end
#*****************************************************************************************************


#*****************************************************************************************************
# Initialization function for creating an initial feasible solution for the production planning problem.
# It attempts to assign each order to the production line that maximizes incremental revenue
# without exceeding the production capacity (H).
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
# Returns:
#      A ProductionPlanningSolution object representing the initial solution with assignments of orders
#      to production lines and their associated revenues.
#*****************************************************************************************************
function Initialization(solver::ProductionPlanningSolver)
    # Initialize an empty solution
    sol = ProductionPlanningSolution(solver.k, solver.H)
    
    # Flags to check if an order is assigned
    order_assigned = falses(solver.size)
    
    # Sequentially try to assign orders to production lines
    for order in 1:solver.size
        best_line = 0
        best_incremental_revenue = -1
        # Evaluate all production lines to find the best fit for the current order
        for line in 1:solver.k
            # Check if the order can be assigned to the current line without exceeding the horizon
            if sum(solver.p[sol.assignment[line]]) + solver.p[order] <= solver.H
                # Calculate the incremental revenue for this order on this line
                incremental_revenue = solver.rev[order]
                # Add revenue from pairwise cost savings with orders already in line
                for assigned_order in sol.assignment[line]
                    incremental_revenue += solver.rev_pair[min(order, assigned_order), max(order, assigned_order)]
                end
                # Update the best line if this line provides a higher revenue
                if incremental_revenue > best_incremental_revenue
                    best_incremental_revenue = incremental_revenue
                    best_line = line
                end
            end
        end
        
        # Assign the order to the best production line if possible
        if best_line > 0
            push!(sol.assignment[best_line], order)
            sol.production_times[best_line] += solver.p[order]
            order_assigned[order] = true
            # Update the objective value of the solution
            sol.objective += best_incremental_revenue
        end
    end
    
    return sol
end


#*****************************************************************************************************
# Function to normalize the reward vector (rho) into a probability distribution for selecting
# destroy methods. Each method's selection probability is directly proportional to its reward value.
# Arguments:
#      rho: A vector representing rewards for different methods.
#      probability: A vector to be filled with the calculated probabilities.
# Returns:
#      The updated probability vector.
#*****************************************************************************************************
function SetProb(rho, probability)
    sum_rho = sum(rho[:])
    for i in eachindex(probability)
        # the higher the reward, the higher the probability
        probability[i] = rho[i] / sum_rho
    end 
    return probability
end


#*****************************************************************************************************
# Function to select a destroy method based on the probability distribution calculated by SetProb.
# It generates a random number to determine the method to be selected using cumulative probabilities.
# Arguments:
#      probability: A normalized vector representing the probability of selecting each destroy method.
# Returns:
#      An integer representing the index of the selected destroy method.
# Raises:
#      An error if the selection process fails, which should theoretically never happen.
#*****************************************************************************************************
function Select_Destroy(probability)
    chosen = rand()
    next_prob = 0
    for i in eachindex(probability)
        next_prob += probability[i]
        if chosen < next_prob
            return i
        end
    end
    println("We should never get here, chosen: ", chosen, " next_prob: ", next_prob)
    exit(0)
end


#*****************************************************************************************************
# Custom sampling function to select K random unique orders from a list of available orders.
# It shuffles the available orders and selects the first K items from this randomized list.
# Arguments:
#      available_orders: A Vector of Int representing the orders that can be selected.
#      K: The number of orders to select.
# Returns:
#      A Vector of Int containing the indices of the K selected orders.
# Raises:
#      An error if the number of samples requested exceeds the number of available orders.
#*****************************************************************************************************
function custom_sample(available_orders::Vector{Int}, K::Int)
    if K > length(available_orders)
        error("Cannot draw more samples than available without replacement.")
    end

    shuffled_orders = shuffle(available_orders)  # Randomly shuffle the array
    return shuffled_orders[1:K]  # Return the first K elements
end


#*****************************************************************************************************
# Function to remove an order from its current assignment in the production planning solution.
# It updates the production time and moves the order to the list of remaining products.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      sol: A ProductionPlanningSolution object representing the current solution.
#      order: An Int representing the order to be removed.
# Side Effects:
#      Updates the production times and assignments within the solution object.
#*****************************************************************************************************
function remove_order!(solver::ProductionPlanningSolver, sol::ProductionPlanningSolution, order)
    for line in 1:solver.k
        if order in sol.assignment[line]
            # Remove the order from the production line.
            deleteat!(sol.assignment[line], findfirst(isequal(order), sol.assignment[line]))
            # Update the production time for the line.
            sol.production_times[line] -= solver.p[order]  # Update production time
            push!(sol.remaining_products, order)  # Move to unassigned products
            break
        end
    end
end


#*****************************************************************************************************
# Apocalypse function simulates a "random destruction" of the current solution by removing a specified
# number of orders from production lines. This method contributes to the diversification of the solution space.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      sol: A ProductionPlanningSolution object representing the current solution.
#      num_orders_to_destroy: An Int64 specifying the number of orders to remove.
# Returns:
#      A ProductionPlanningSolution object with the specified number of orders removed.
# Side Effects:
#      Can print a warning message if not enough orders are available for destruction.
#*****************************************************************************************************
function Apocalypse!(solver::ProductionPlanningSolver, sol::ProductionPlanningSolution, num_orders_to_destroy::Int64)
    # Ensure we do not attempt to remove more orders than are currently assigned.
    num_orders_to_destroy = min(num_orders_to_destroy, solver.size - length(sol.remaining_products))

    # Collect all orders that are currently assigned to any production line
    assigned_orders = [order for line in sol.assignment for order in line]
    # These are the orders that can potentially be removed
    available_orders = assigned_orders

    # Use the custom_sample function to select orders to remove
    if length(available_orders) < num_orders_to_destroy
        println("Not enough orders available to destroy: Needed ", num_orders_to_destroy, ", Available: ", length(available_orders))
        return sol  # If not enough orders are available to destroy, exit early
    end

    # Randomly pick orders to remove, ensuring they are not already in the rem_products
    orders_to_remove = custom_sample(Int64.(available_orders), num_orders_to_destroy)

    # Iterate over each selected order and remove it from its production line.
    for order in orders_to_remove
        removed = false  # Flag to check if order has been removed
        for line in 1:solver.k
            if order in sol.assignment[line]
                # Remove the order from the production line.
                index = findfirst(isequal(order), sol.assignment[line])
                deleteat!(sol.assignment[line], index)
                # Update the production time for the line.
                sol.production_times[line] -= solver.p[order]
                # Add the order to the list of unassigned products.
                push!(sol.remaining_products, order)
                removed = true
                # Break the inner loop since we've found and removed the order.
                break
            end
        end
        if !removed
            println("Order ", order, " not found in any production line, which should not happen.")
        end
    end
    return sol
end


#*****************************************************************************************************
# Function to create a neighborhood matrix based on revenue impact, which will be used in determining
# the closest neighbors for a given order based on potential revenue increase.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
# Returns:
#      A Matrix representing the neighborhood relationships based on revenue impact.
#*****************************************************************************************************
function create_Revenue_Based_NeighbourMatrix(solver::ProductionPlanningSolver)
    # Initialize the neighborhood matrix
    neigh = zeros(Int, solver.size, solver.size)

    # Populate the matrix based on revenue potential
    for i in 1:solver.size
        # Sort orders based on total potential revenue when paired with order i
        revenue_impacts = [(j, solver.rev[i] + solver.rev[j] + solver.rev_pair[i, j]) for j in 1:solver.size if i != j]
        sort!(revenue_impacts, by = x -> x[2], rev = true)  # Highest revenue impact first

        # Assign rankings based on revenue impact
        for rank in eachindex(revenue_impacts)
            j, _ = revenue_impacts[rank]
            neigh[i, j] = rank
        end
    end

    return neigh
end


#*****************************************************************************************************
# Ragnarok function applies a targeted destruction to the current solution based on a set of orders
# and their nearest neighbors, as defined by a neighborhood matrix. It is designed to disrupt the solution
# in a way that allows the ALNS algorithm to explore more varied regions of the solution space.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      sol: A ProductionPlanningSolution object representing the current solution.
#      neigh: A Matrix representing neighborhood relationships.
#      K: The number of primary orders to select for removal.
#      L: The number of nearest neighbors to also remove.
# Returns:
#      The modified ProductionPlanningSolution object after the destruction process.
#*****************************************************************************************************
function Ragnarok!(solver::ProductionPlanningSolver, sol::ProductionPlanningSolution, neigh::Matrix{Int64}, K::Int64, L::Int64)
    # Get the list of orders not currently remaining (i.e., those that are active in the solution)
    active_orders = setdiff(1:solver.size, sol.remaining_products)
    
    # Ensure we do not attempt to remove more orders than are available
    K = min(K, length(active_orders))
    
    if K == 0
        return sol  # If no orders can be removed, return the solution unchanged
    end

    selected_orders = custom_sample(active_orders, K)
    destroyed = Set{Int32}()
    for order in selected_orders
        push!(destroyed, order)
        # Get L nearest neighbors according to the neighborhood matrix
        neighbor_orders = sortperm(neigh[order, :], by=x->x, rev=true)[1:L]
        for neighbor in neighbor_orders
            push!(destroyed, neighbor)
            if length(destroyed) >= solver.size
                break
            end
        end
    end
    # Remove all selected orders from their respective production lines
    for order in destroyed
        remove_order!(solver, sol, order)
    end
    return sol
end


#*****************************************************************************************************
# Calculates the total revenue after a destructor function has been applied to the solution.
# It iterates through each production line and computes the sum of individual revenues
# and additional revenues from pairing orders on the same line.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      sol: A ProductionPlanningSolution object representing the current partial solution.
# Returns:
#      An Int representing the total revenue of the solution after destruction.
#*****************************************************************************************************
function post_destructor_objective(solver::ProductionPlanningSolver, sol::ProductionPlanningSolution)
    total_revenue = 0
    for line in 1:solver.k
        line_revenue = 0
        for i in 1:length(sol.assignment[line])
            order = sol.assignment[line][i]
            line_revenue += solver.rev[order]  # add individual revenue
            # Add pairwise revenue with other orders on the same line
            for j in (i+1):length(sol.assignment[line])
                other_order = sol.assignment[line][j]
                line_revenue += solver.rev_pair[min(order, other_order), max(order, other_order)]
            end
        end
        total_revenue += line_revenue
    end
    return total_revenue  # return the calculated revenue
end


#*****************************************************************************************************
# Inserts an order into a specified production line and updates the solution's state.
# It modifies the assignment array, updates production times, and recalculates the objective.
# Arguments:
#      sol: A ProductionPlanningSolution object representing the current solution.
#      order: An Int representing the order to be inserted.
#      line: An Int representing the production line where the order will be inserted.
#      solver: A ProductionPlanningSolver object containing problem-specific data.
# Side Effects:
#      Updates the assignment and production times within the solution object and increments the objective.
#*****************************************************************************************************
function insert_order!(sol, order, line, solver)
    push!(sol.assignment[line], order)
    deleteat!(sol.remaining_products, findfirst(isequal(order), sol.remaining_products))
    sol.production_times[line] += solver.p[order]
    sol.objective += calculate_incremental_revenue(sol, order, line, solver)
end


#*****************************************************************************************************
# Calculates the incremental revenue of inserting an order into a production line.
# This function is called during the repair process to evaluate the best insertion.
# Arguments:
#      sol: A ProductionPlanningSolution object representing the current solution.
#      order: An Int representing the order to be potentially inserted.
#      line: An Int representing the production line for the insertion evaluation.
#      solver: A ProductionPlanningSolver object containing problem-specific data.
# Returns:
#      An Int representing the incremental revenue of inserting the order into the line.
#*****************************************************************************************************
function calculate_incremental_revenue(sol, order, line, solver)
    incremental_revenue = solver.rev[order]
    for assigned_order in sol.assignment[line]
        if assigned_order != order
            incremental_revenue += solver.rev_pair[min(order, assigned_order), max(order, assigned_order)]
        end
    end
    return incremental_revenue
end


#*****************************************************************************************************
# The repair function, nicknamed "Bob The Builder," that reconstructs a feasible solution
# from the destroyed state. It seeks to maximize revenue by choosing the best insertions
# of remaining products into the production lines.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      sol: A ProductionPlanningSolution object representing the current partial solution.
# Returns:
#      A ProductionPlanningSolution object representing the repaired solution.
# Side Effects:
#      Modifies the solution by inserting orders and updating the objective incrementally.
#*****************************************************************************************************
function Bob_The_Builder!(solver::ProductionPlanningSolver, sol::ProductionPlanningSolution)
    # Reset or reinitialize the objective if not handled during destroy
    sol.objective = post_destructor_objective(solver, sol)  # This calculates based on the current partial solution post-destroy

    while !isempty(sol.remaining_products)
        # Similar insertion logic as previously described
        best_increase = -Inf
        best_order_to_insert = nothing
        best_line_for_order = 0
        for order in sol.remaining_products
            for line in 1:solver.k
                if sum(solver.p[sol.assignment[line]]) + solver.p[order] <= solver.H
                    incremental_revenue = solver.rev[order]
                    for assigned_order in sol.assignment[line]
                        incremental_revenue += solver.rev_pair[min(order, assigned_order), max(order, assigned_order)]
                    end
                    if incremental_revenue > best_increase
                        best_increase = incremental_revenue
                        best_order_to_insert = order
                        best_line_for_order = line
                    end
                end
            end
        end
        if best_order_to_insert === nothing
            break
        end
        insert_order!(sol, best_order_to_insert, best_line_for_order, solver)
    end
    return sol
end


#*****************************************************************************************************
# Adaptive Large Neighborhood Search (ALNS) algorithm for solving the production planning problem.
# It iteratively destroys and repairs the solution, seeking to improve the total revenue.
# Arguments:
#      solver: A ProductionPlanningSolver object containing problem-specific data.
#      timeLimit: An integer representing the time limit for the search process in nanoseconds.
# Returns:
#      A ProductionPlanningSolution object representing the best solution found within the time limit.
# Process:
#      - Initializes a solution and iteratively applies destruction and repair methods.
#      - Uses a probabilistic method to select between different destroy strategies.
#      - Implements a simulated annealing-like acceptance criterion to escape local optima.
#      - Reheats the simulated annealing process after a certain number of consecutive acceptances.
#      - Updates a reward system to adjust the probabilities of selecting destroy methods.
#      - Returns the best solution found after completing the search within the time limit.
# Side Effects:
#      Can print progress messages to the console and modify global state related to timer and probability vectors.
#*****************************************************************************************************
function ALNS(solver::ProductionPlanningSolver, timeLimit::Int)
    # Set timer and neighborhood
    startTime = time_ns()
    neigh = create_Revenue_Based_NeighbourMatrix(solver)

    current_solution = Initialization(solver)
    best_solution = deepcopy(current_solution)

    min_delete::Int64 = 2
    max_delete::Int64 = ceil(solver.size*0.3)
    if max_delete < min_delete
        max_delete = min_delete + 1 
    end
    println("Destroy interval: [", min_delete, " , ", max_delete, "]")

    it = 1

    rho = ones(2) # reward vector
    probability = zeros(2) # probability vector
    probability = SetProb(rho, probability) # Set initial probability

    # Reward constants for global_best / better than current / accepted / not accepted
    W1, W2, W3, W4 = 10, 5, 1, 0

    gamma = 0.9 # decay constant
    T = 1000 # initial temperature
    alpha = 0.999 # temperature decay parameter

    sa_accept_counter = 0          # Initialize the counter for SA acceptances
    sa_accept_threshold = 4000     # Set the threshold for consecutive acceptances
    max_temperature = 10000

    while elapsedTime(startTime) < timeLimit

        # Update the probability vector in batches of 10 iterations
        if mod(it,10) == 0 
            probability = SetProb(rho, probability)
        end

        # Create temporary_sol
        temporary_solution = deepcopy(current_solution)

        # Pick the destroy method
        selected_destroy_method  = Select_Destroy(probability)
        if selected_destroy_method == 1
            num_cities_to_destroy = rand(min_delete:max_delete)
            temporary_solution = Apocalypse!(solver,temporary_solution,num_cities_to_destroy)
        else
            # Set K and L based on a percentage of the total number of orders
            K = max(1, ceil(Int64, 0.1 * solver.size))  # At least 1
            L = max(1, ceil(Int64, 0.1 * solver.size))  # At least 1
            temporary_solution = Ragnarok!(solver,temporary_solution,neigh,K,L)
        end

        # the objective is re-calculated in here
        temporary_solution = Bob_The_Builder!(solver,temporary_solution)

        # initialize not accepted score
        score = W4
        if temporary_solution.objective > current_solution.objective
            # improving the current solution
            current_solution = temporary_solution
            score = W2 
            sa_accept_counter = 0 # Reset the counter because we found a better solution
        else
            # go with simulated annealing acceptance criterio
            if rand() < exp(-(temporary_solution.objective - current_solution.objective)/T)
                current_solution = temporary_solution
                score = W3
                sa_accept_counter += 1 #Increment the counter for each SA acceptance
            end
        end

        if temporary_solution.objective > best_solution.objective
            best_solution = deepcopy(temporary_solution)
            score = W1
        end

        # Check for the need to reheat
        if sa_accept_counter >= sa_accept_threshold && isclose(current_solution.objective, best_solution.objective, atol= 0.01)
            println("Reheating applied at iteration ", it)
            T =  randn() * max_temperature # Apply the reheat
            sa_accept_counter = 0 # Reset the counter
        end

        # update the reward vectors for the selected destroy method
        rho[selected_destroy_method] = gamma*rho[selected_destroy_method] + (1-gamma)*score

        # report progress made
        if mod(it,100) == 0
            println("it: ", it, " time: ", elapsedTime(startTime), " < ", timeLimit, " objective: ", best_solution.objective, "prob: ", probability, " rho: ", rho)
        end

        # re-calculate temperature
        T *= alpha
        it += 1
    end
    return best_solution
end


#*****************************************************************************************************
# Writes the best solution's assignment of orders to production lines to a file. This function 
# iterates through each production line in the solution, converts the orders to a string, and 
# writes them to the specified file, one line per production line.
# Arguments:
#      file: The file handle to write the solution.
#      solution: A ProductionPlanningSolution object containing the assignment of orders to lines.
#*****************************************************************************************************
function write_solution_to_file(file, solution)
    for orders in solution.assignment
        # Convert the orders array to a string representation
        orders_str = join(orders, " ")  # Joins the order IDs with spaces
        write(file, orders_str, "\n")  # Write each line's orders to the file, followed by a newline
    end
end


#*****************************************************************************************************
# The main function is the entry point of the program. It parses command-line arguments to get the instance file, solution file, and time limit for the ALNS. It then proceeds 
# to read the instance, execute the ALNS to find the best solution, and writes this solution to the 
# solution file provided as an argument.
#*****************************************************************************************************
function main()
    # Check if the correct number of arguments are passed
    if length(ARGS) != 3
        println("Usage: julia s222717.jl <instance_file> <solution_file> <time_limit>")
        return
    end

    # Parse command-line arguments
    filename = ARGS[1]
    solution_file = ARGS[2]
    time_limit = parse(Int, ARGS[3])
    name, size, LB, rev, rev_pair, k, H, p = read_instance(filename)

    # Create a ProductionPlanningSolver instance with the data
    solver = ProductionPlanningSolver(rev, rev_pair, p, k, H, size)

    best_solution = ALNS(solver,time_limit)
    println("Best solution found with objective value: ", best_solution.objective)
    println("Best solution assignment to production lines:")
    for (line, orders) in enumerate(best_solution.assignment)
        println("Line ", line, ": ", orders)
    end

    # Write the best solution to the solution file
    open(solution_file, "w") do file
        write_solution_to_file(file, best_solution)
    end
end
#*****************************************************************************************************


#*****************************************************************************************************
# Run the main function if this script is executed as a program
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
#*****************************************************************************************************