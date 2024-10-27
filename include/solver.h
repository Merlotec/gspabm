//
// Created by ncbmk on 10/22/24.
//
// Updated documentation and comments for clarity.
//

#ifndef SOLVER_H
#define SOLVER_H

#include "peris.h"
#include "render.h"
#include <cassert>
#include <optional>

#ifndef __ssize_t_defined
typedef long int ssize_t;
#endif

#define BUFFER_MARGIN_FACTOR 10.f

namespace peris {

    /// Enumeration of possible results from the solver functions
    enum SolutionResult : int {
        err_cycle = -4,
        err_budget_constraint = -3,
        err_nan = -2,
        err_unknown = -1,
        terminated = 0,
        success = 1,
        repeat = 2,
    };

    /// @brief The Solver class calculates Pareto efficient allocations of items to agents.
    ///        Each agent receives exactly one item. The goal is to determine prices such that
    ///        no agent would prefer any other agent's allocation at the given prices.
    ///
    /// @tparam A The agent type, must satisfy AgentConcept (e.g., have income(), utility() methods)
    /// @tparam I The item type, must satisfy ItemConcept (e.g., have quality() method)
    ///
    template<typename A, typename I>
        requires AgentConcept<A> && ItemConcept<I>
    class Solver {
        /// Vector containing the current allocations of items to agents
        /// Each Allocation contains an agent, item, price, and utility
        std::vector<Allocation<A, I> > allocations;

        /// @brief Swaps the agents between allocations at indices a and b,
        ///        and recalculates the utility for the affected allocations.
        ///
        /// @param a Index of the first allocation
        /// @param b Index of the second allocation
        ///
        void swap_agents(size_t a, size_t b) {
            auto agent_a = allocations[a].agent;
            auto agent_b = allocations[b].agent;

            // Swap the agent objects.
            allocations[a].agent = agent_b;
            // Calculate utility for new agent.
            allocations[a].recalculate_utility();

            // Repeat for other agent
            allocations[b].agent = agent_a;
            allocations[b].recalculate_utility();
        }

        /// @brief Moves the agent at index 'a' to index 'b' in the allocations vector,
        ///        shifting agents between positions 'b' and 'a-1' up by one position.
        ///        This effectively inserts the agent at position 'a' into position 'b',
        ///        pushing other agents forward in the vector.
        ///
        /// @param a Source index of the agent to move (must be greater than b)
        /// @param b Destination index where the agent will be placed
        ///
        void displace_up(size_t a, size_t b) {
            assert(b < a); // Ensure that the source index 'a' is greater than the destination index 'b'.

            // Temporarily store the agent at position 'b' as it will be overridden.
            auto free_agent = allocations[b].agent;

            // Move the agent from position 'a' to position 'b'.
            allocations[b].agent = allocations[a].agent;
            allocations[b].recalculate_utility(); // Update utility after changing the agent.

            // Shift agents from position 'b+1' to 'a' up by one position.
            // This loop moves each agent into the position of the previous agent.
            for (size_t i = b + 1; i <= a; ++i) {
                // Store the current agent to be moved in the next iteration.
                auto agent_buffer = allocations[i].agent;

                // Move the 'free_agent' into the current position.
                allocations[i].agent = free_agent;
                allocations[i].recalculate_utility(); // Update utility after changing the agent.

                // Update 'free_agent' for the next iteration.
                free_agent = agent_buffer;
            }
            // After the loop, 'free_agent' holds the agent originally at position 'a',
            // which has already been moved to position 'b', so it can be discarded.
        }

        /// @brief Moves the agent at index 'a' to index 'b' in the allocations vector,
        ///        shifting agents between positions 'a+1' and 'b' down by one position.
        ///        This effectively inserts the agent at position 'a' into position 'b',
        ///        pulling other agents backward in the vector.
        ///
        /// @param a Source index of the agent to move (must be less than b)
        /// @param b Destination index where the agent will be placed
        ///
        void displace_down(size_t a, size_t b) {
            assert(b > a); // Ensure that the destination index 'b' is greater than the source index 'a'.

            // Temporarily store the agent at position 'b' as it will be overridden.
            auto free_agent = allocations[b].agent;

            // Move the agent from position 'a' to position 'b'.
            allocations[b].agent = allocations[a].agent;
            allocations[b].recalculate_utility(); // Update utility after changing the agent.

            // Shift agents from position 'b-1' down to 'a' by one position.
            // This loop moves each agent into the position of the next agent.
            for (ssize_t i = b - 1; i >= (ssize_t)a; --i) {
                // Store the current agent to be moved in the next iteration.
                auto agent_buffer = allocations[i].agent;

                // Move the 'free_agent' into the current position.
                allocations[i].agent = free_agent;
                allocations[i].recalculate_utility(); // Update utility after changing the agent.

                // Update 'free_agent' for the next iteration.
                free_agent = agent_buffer;
            }
            // After the loop, 'free_agent' holds the agent originally at position 'a',
            // which has already been moved to position 'b', so it can be discarded.
        }

        double calculate_efficient_price(const Allocation<A, I> &l, const Allocation<A, I> &a, const double epsilon, const int max_iter) {

            double max_price = l.agent.income() - epsilon;
            double min_price = l.price == 0 ? epsilon : l.price - epsilon;

            double efficient_price = indifferent_price(l.agent, a.quality(), l.utility,
                                                min_price, max_price, epsilon, max_iter);

            if (isnan(efficient_price)) {
                // If unable to compute efficient price, use max price if it improves utility
                if (l.agent.utility(max_price, a.quality()) > l.utility) {
                    efficient_price = max_price;
                }
            }
            return efficient_price;
        }

    public:
        ///
        /// @brief Constructs the Solver with given agents and items.
        ///
        ///        - Asserts that the number of agents equals the number of items.
        ///        - Sorts agents in order of increasing income.
        ///        - Sorts items in order of increasing quality.
        ///        - Initializes Allocations vector pairing each agent with an item,
        ///          setting an initial guess price based on the agent's income.
        ///
        /// @param agents Vector of agents
        /// @param items  Vector of items
        /// @param guess_factor Initial guess factor for prices (used as price = guess_factor * agent.income())
        ///
        Solver(std::vector<A> agents, std::vector<I> items, double guess_factor) {
            // Ensure that there is one item per agent (numbers of each are the same).
            assert(agents.size() == items.size());

            // Sort agents by income (increasing order), so lower-income agents are first.
            std::sort(agents.begin(), agents.end(), [](A a, A b) { return a.income() < b.income(); });
            // Sort items by quality (increasing order), so lower-quality items are first.
            std::sort(items.begin(), items.end(), [](I a, I b) { return a.quality() < b.quality(); });

            allocations.reserve(items.size());
            // Combine the items and agents into allocations.
            for (size_t i = 0; i < items.size(); i++) {
                // Set the initial guess price to an arbitrary guess according to the function p_i = guess_factor * y_i
                const double guess_price = guess_factor * agents[i].income();
                Allocation<A, I> allocation = {
                    .item = items[i],
                    .agent = agents[i],
                    .price = guess_price,
                    .utility = agents[i].utility(guess_price, items[i].quality())
                };

                allocations.push_back(allocation);
            }
        }

        ///
        /// @brief Returns a copy of the current allocations vector.
        ///
        /// @return A vector of Allocations containing the current allocation state.
        ///
        std::vector<Allocation<A, I>> get_allocations() {
            return std::vector<Allocation<A, I>>(this->allocations);
        }

        ///
        /// @brief Solves the allocation model to achieve Pareto efficiency among agents.
        ///        The algorithm assigns items to agents in a way that no agent can be made better off
        ///        without making another agent worse off. It iteratively adjusts allocations and prices,
        ///        possibly swapping agents to improve overall efficiency.
        ///
        /// @param render_state Pointer to a RenderState object for visualization (can be nullptr)
        /// @param epsilon The tolerance for numerical approximations (default is 1e-5)
        /// @param max_iter The maximum number of iterations for convergence (default is 200)
        ///
        /// @return A SolutionResult indicating success or type of error.
        ///
        SolutionResult solve(RenderState<A, I>* render_state, double epsilon = 1e-5, int max_iter = 200) {
            // If there are no agents, return success.
            if (allocations.empty()) {
                return SolutionResult::success;
            }

            // Start alignment process from index 0
            return align(render_state, 0, epsilon, max_iter);
        }

    private:
        ///
        /// @brief Finds the index of the most preferred allocation for agent at index 'i' among allocations.
        ///
        ///        It goes through other allocations and computes the utility the agent at index 'i' would get
        ///        from those allocations (after adjusting the price slightly to avoid division by zero or
        ///        price equality issues). It returns the index of the allocation that provides the maximum utility.
        ///
        /// @param i Index of the agent/allocation to consider.
        /// @param search_above If true, search only for allocations with higher indices than 'i'
        /// @param epsilon Tolerance used for price adjustments
        ///
        /// @return The index of the most preferred allocation for agent at index 'i'
        ///
        size_t most_preferred(size_t i, bool search_above, double epsilon) {
            assert(i < allocations.size());
            double u_max;
            if (allocations[i].agent.income() < allocations[i].price) {
                u_max = allocations[i].agent.utility(allocations[i].price, allocations[i].item.quality());
            } else {
                u_max = NAN;
            }
            size_t i_max = i;

            size_t limit = search_above ? allocations.size() : i;

            // Iterate over allocations to find the most preferred one for agent i
            for (size_t j = 0; j < limit; ++j) {
                if (i != j) {
                    // Compute the utility of agent i if they were allocated allocation j
                    const double u_alt = allocations[i].agent.utility(allocations[j].price, allocations[j].item.quality());
                    if (u_alt > u_max + epsilon || (isnan(u_max) && !isnan(u_alt))) {
                        u_max = u_alt;
                        i_max = j;
                    }
                }
            }

            return i_max;
        }

        ///
        /// @brief Aligns the allocations starting from index 'i' to achieve Pareto efficiency.
        ///
        ///        The align function is the core of the algorithm. It assigns allocations and adjusts prices
        ///        such that each agent prefers their own allocation over any other allocation,
        ///        i.e., no agent would prefer another allocation at the given prices.
        ///
        ///        It may involve adjusting prices, displacing agents (moving them up or down in the allocation list),
        ///        to ensure that allocations are Pareto efficient.
        ///
        /// @param render_state Pointer to a RenderState for visualization (can be nullptr)
        /// @param i Starting index for alignment
        /// @param epsilon Numerical tolerance for calculations
        /// @param max_iter Maximum number of iterations for numerical methods
        ///
        /// @return A SolutionResult indicating success or type of error.
        ///
        SolutionResult align(RenderState<A, I>* render_state, size_t i, double epsilon, int max_iter = 100) {
            // Initialize the first allocation if starting from index 0
            if (i == 0) {
                // Set the price of the first allocation to zero.
                // This essentially 'anchors' the algorithm so that all other allocations can be distributed around this value.
                allocations[0].set_price(0.0f);

                // We have allocated agent 0 so we can start at agent 1.
                i = 1;
            }

            // Keeps track of the highest index that has so far been reached.
            // This is used for rendering so that we only re-render when the next allocation has been reached.
            size_t head = i;

            // Iterate through each agent starting from index i
            while (i < allocations.size()) {
                // Visualization code if render_state is provided
                if (render_state != nullptr) {
                    if (i > head) {
                        if (!render_state->draw_allocations(this->allocations, i)) {
                            return SolutionResult::terminated;
                        }
                    }
                }
                head = max(i, head);

                // Initialize variables to keep track of agents to displace or promote
                ssize_t agent_to_displace = -1;
                ssize_t agent_to_promote = -1;

                Allocation<A, I>& a = allocations[i];     // Current allocation
                Allocation<A, I>& l = allocations[i - 1]; // Previous allocation

                double efficient_price = calculate_efficient_price(l, a, epsilon, max_iter);

                // Check if the efficient price exceeds the current agent's income
                if (efficient_price + epsilon > a.agent.income()) {
                    // Find the most preferred allocation for the current agent among those below
                    size_t new_i = most_preferred(i, false, epsilon); // Do not search above because not yet allocated.
                    agent_to_displace = new_i;
                }

                if (agent_to_displace == -1) {

                    for (ssize_t j = i - 1; j >= 0; --j) {
                        const Allocation<A, I>& prev = allocations[j];
                        // Check if the previous agent prefers the current allocation at efficient price
                        if (prev.agent.income() > efficient_price) {
                            if (j < i - 1 && prev.agent.utility(efficient_price, a.quality()) > prev.utility + epsilon) {
                                // Mark this agent to promote
                                agent_to_promote = j;

                                // Calculate the new efficient price by making this 'prev' agent indifferent.
                                double new_price = calculate_efficient_price(prev, a, epsilon, max_iter);

                                if (new_price > efficient_price) {
                                    efficient_price = new_price;
                                }
                            }
                        }
                    }

                    // Check if the efficient price exceeds the current agent's income
                    if (efficient_price + epsilon > a.agent.income()) {
                        // Find the most preferred allocation for the current agent among those below
                        size_t new_i = most_preferred(i, false, epsilon); // Do not search above because not yet allocated.
                        agent_to_displace = new_i;
                    }

                    if (agent_to_displace == -1) {
                        // Check if this agent prefers any previous allocations
                        double u_max = a.agent.utility(efficient_price, a.quality());

                        if (isnan(u_max))
                            return SolutionResult::err_nan;
                        for (ssize_t j = i - 1; j >= 0; --j) {
                            const Allocation<A, I>& prev = allocations[j];
                            // Check if the agent can afford the previous allocation
                            if (a.agent.income() > prev.price + epsilon) {
                                double u_prev = a.agent.utility(prev.price, prev.quality());
                                if (isnan(u_prev))
                                    return SolutionResult::err_nan;
                                if (u_prev > u_max + epsilon) {
                                    // The current agent 'a' prefers 'prev''s allocation; mark 'prev' as the agent to displace.
                                    u_max = u_prev;
                                    agent_to_displace = j;
                                }
                            }
                        }

                        // If no displacement is needed, update the current allocation's price and utility.
                        if (agent_to_displace == -1) {
                            if (efficient_price + epsilon > a.agent.income()) {
                                // Find best place to move agent to.
                                size_t new_i = most_preferred(i, false, epsilon); // Do not search above because not yet allocated.
                                agent_to_displace = new_i;
                            } else {
                                // Update the allocation with the efficient price and utility
                                double efficient_utility = a.agent.utility(efficient_price, a.quality());
                                if (isnan(efficient_utility))
                                    return SolutionResult::err_nan;

                                a.price = efficient_price;
                                a.utility = efficient_utility;
                            }
                        }
                    }
                }

                if (agent_to_displace >= 0) {
                    // Handle displacement or promotion of agents
                    if (agent_to_promote >= 0) {
                        // Displace the agent to the reserve area
                        displace_down(agent_to_promote, allocations.size() - 1);
                        i = max(agent_to_promote, 1L);
                    } else {
                        assert(agent_to_displace < i); // The agent to displace should be at a lower index.

                        // Displace the current agent 'a' to position 'agent_to_displace', shifting other agents accordingly.
                        displace_up(i, agent_to_displace);
                        i = max(agent_to_displace, 1L);
                    }
                } else {
                    // The current allocation is successful, so we can move on to the next one.
                    ++i;
                }
            }
            return SolutionResult::success;
        }

        ///
        /// @brief Adjusts prices to ensure that no agent prefers another allocation.
        ///
        ///        The push method iteratively increases prices where necessary to prevent agents
        ///        from preferring allocations assigned to other agents. It ensures that the prices
        ///        are set such that each agent's utility from their own allocation is at least as
        ///        high as any other allocation they could afford.
        ///
        /// @param epsilon Numerical tolerance for calculations
        /// @param max_iter Maximum number of iterations for numerical methods
        ///
        /// @return A SolutionResult indicating whether any prices were updated or an error occurred.
        ///
        SolutionResult push(double epsilon, int max_iter) {
            size_t updated = 0;
            // Start from the top allocation and move backwards
            for (ssize_t i = allocations.size() - 1; i >= 0; --i) {
                Allocation<A, I>& a = allocations[i];     // Current allocation
                double efficient_price = a.price;

                // Check if any earlier agents prefer the current allocation at 'efficient_price'
                for (ssize_t j = allocations.size() - 1; j >= 0; --j) {
                    if (i == j) {
                        continue;
                    }
                    Allocation<A, I>& other = allocations[j];

                    // Ensure that 'other' can afford the current allocation
                    if (efficient_price + epsilon < other.agent.income()) {
                        // If agent 'other' prefers the current allocation at 'efficient_price'
                        if (other.agent.utility(efficient_price, a.quality()) > other.utility + epsilon) {
                            // Update the efficient price to prevent 'other' from preferring it
                            double max_price = other.agent.income() - epsilon;
                            double min_price = other.price == epsilon;

                            double new_price = indifferent_price(other.agent, a.quality(), other.utility, min_price, max_price, epsilon, max_iter);
                            if (isnan(new_price)) {
                                if (other.agent.utility(max_price, a.quality()) > other.utility) {
                                    efficient_price = max_price;
                                } else {
                                    return SolutionResult::err_nan;
                                }
                            } else if (new_price > efficient_price) {
                                efficient_price = new_price;
                            }
                        }
                    }
                }

                // Update to reflect new efficient price
                if (efficient_price > a.price + epsilon) {
                    // Update efficient price and utility
                    if (efficient_price + epsilon > a.agent.income()) {
                        return SolutionResult::err_budget_constraint;
                    }
                    a.set_price(efficient_price);
                    ++updated;
                }
            }

            if (updated > 0) {
                return SolutionResult::repeat;
            } else {
                return SolutionResult::success;
            }
        }

    public:
        ///
        /// @brief Verifies that the current allocation is a valid solution.
        ///
        ///        Checks that no agent would prefer any other agent's allocation at the given prices.
        ///        Ensures that each agent's utility is correct and that no agent can improve their utility
        ///        by switching to another allocation they can afford.
        ///
        /// @param epsilon Numerical tolerance for comparisons (default is 1e-6)
        ///
        /// @return True if the current allocation is valid; false otherwise.
        ///
        bool verify_solution(const double epsilon = 1e-6) const {
            for (size_t i = 0; i < allocations.size(); ++i) {
                double u = allocations[i].agent.utility(allocations[i].price, allocations[i].item.quality());
                if (u != allocations[i].utility) {
                    std::cout << "Agent " << i << " has utility mismatch!" << std::endl;
                    return false;
                }

                for (size_t j = 0; j < allocations.size(); ++j) {
                    if (i != j) {
                        if (allocations[j].agent.item_id() == allocations[i].agent.item_id()) {
                            std::cout << "Agent " << i << " has the same item_id as " << j << "; item_id= " << allocations[j].agent.item_id() << std::endl;
                            return false;
                        }
                        // Compute the utility agent i would get from allocation j
                        const double u_alt = allocations[i].agent.utility(allocations[j].price + 2.f * epsilon, allocations[j].item.quality());
                        if (u_alt > u) {
                            std::cout << "Agent " << i << " prefers allocation " << j << "; " << u_alt << ">" << u << std::endl;
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        ///
        /// @brief Draws the current allocations using the provided RenderState.
        ///
        /// @param render_state Pointer to a RenderState object for visualization.
        ///
        /// @return True if drawing was successful; false if terminated.
        ///
        bool draw(RenderState<A, I>* render_state) {
            return render_state->draw_allocations(this->allocations, -1);
        }

        ///
        /// @brief Performs a linear regression of price on quality.
        ///
        ///        Calculates the best-fit line of the form price = a + b * quality
        ///        using least squares regression, and outputs the regression coefficients
        ///        and the coefficient of determination (R^2).
        ///
        void regress_price_on_quality() {
            // Ensure there are enough data points
            if (allocations.size() < 2) {
                std::cerr << "Not enough data points to perform regression." << std::endl;
                return;
            }

            size_t n = allocations.size();
            double sum_x = 0.0f;   // Sum of qualities
            double sum_y = 0.0f;   // Sum of prices
            double sum_xx = 0.0f;  // Sum of qualities squared
            double sum_xy = 0.0f;  // Sum of quality * price

            // Calculate sums required for regression coefficients
            for (const auto& alloc : allocations) {
                double x = alloc.quality();
                double y = alloc.price;
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }

            double x_bar = sum_x / n;
            double y_bar = sum_y / n;

            double Sxy = sum_xy - n * x_bar * y_bar;
            double Sxx = sum_xx - n * x_bar * x_bar;

            if (Sxx == 0.0f) {
                std::cerr << "Cannot compute regression coefficients; division by zero." << std::endl;
                return;
            }

            double b = Sxy / Sxx;          // Slope
            double a = y_bar - b * x_bar;  // Intercept

            std::cout << "Regression result: price = " << a << " + " << b << " * quality" << std::endl;

            // Calculate the coefficient of determination (R^2)
            double ss_tot = 0.0f;
            double ss_res = 0.0f;
            for (const auto& alloc : allocations) {
                double x = alloc.quality();
                double y = alloc.price;
                double y_pred = a + b * x;
                ss_tot += (y - y_bar) * (y - y_bar);
                ss_res += (y - y_pred) * (y - y_pred);
            }

            double r_squared = 1 - (ss_res / ss_tot);
            std::cout << "Coefficient of determination (R^2): " << r_squared << std::endl;
        }
    };
}

#endif //SOLVER_H