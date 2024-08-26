# ProductionPlanningUsing_ALNS



This script encapsulates the implementation of an Adaptive Large Neighborhood Search (ALNS) for addressing the production planning problem at PlastOut A/S. The primary objective is to maximize the revenue by effectively planning the assignment of orders to production lines within a specified time horizon, accounting for individual product revenues and additional revenue from pairing products on the same production line.

### The ALNS employs the following strategies and components:
-   **Initialization**: Generates a feasible initial solution that respects production line capacities and maximizes initial revenue using a deterministic heuristic.
-   **Destroy Operators**: Two methods are implemented to remove orders from the production lines, promoting solution space exploration:

                     1. Apocalypse - Randomly removes a subset of orders. 2. Ragnarok - Targets specific orders and their neighbors for removal.

-   **Repair Operator**: A greedy heuristic, Bob_The_Builder, that reinserts orders to optimize the current solution, aiming for revenue increment.
-   **Adaptive Mechanism:** Dynamically adjusts the probability of employing destroy operators based 
                       on their past performance, balancing diversification and intensification.
                       Reheating is applied for the simulated annealing after an extended period of iterations.
-   **Acceptance Criterion**: A combination of Improving Solution and a Simulated Annealing acceptance strategy is utilized to escape local optima by accepting non-improving moves with a certain probability.
-   **Termination Criteria**: The algorithm concludes when a predefined computational time has been reached, ensuring efficient execution.
-   **Objective Calculation**: Revenue calculation considers both individual order revenues and pairwise revenue boosts for co-located orders.
