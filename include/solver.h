//
// Created by ncbmk on 10/22/24.
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

namespace peris {
      /// The class providing the solving functionality, requiring an agent type A and a utility function, U.
    /// Templates are used so that the function can be inlined, and so that the agent type can be stored in the array.
    /// This improves cache locality, which should improve performance.
    template<typename A, typename I>
        requires AgentConcept<A> && ItemConcept<I>
    class Solver {
        /// Describes what offer is allocated to what agent. Since the item to be allocated always remains in the same
        /// allocation object, the index of the allocation uniquely identifies the item.
        std::vector<Allocation<A, I> > allocations;

        void swap_agents(size_t a, size_t b) {
            auto agent_a = allocations[a].agent;
            auto agent_b = allocations[b].agent;

            // Swap the agent object.
            allocations[a].agent = agent_b;
            // Calculate utility for new agent.
            allocations[a].recalculate_utility();

            // Repeat for other agent
            allocations[b].agent = agent_a;
            allocations[b].recalculate_utility();
        }

        // Moves the agent at index 'a' to index 'b' in the allocations vector,
        // shifting agents between positions 'b' and 'a-1' up by one position.
        // This effectively inserts the agent at position 'a' into position 'b',
        // pushing other agents forward in the vector.
        void displace(size_t a, size_t b) {
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

    public:
        Solver(std::vector<A> agents, std::vector<I> items, float guess_factor) {
            // Ensure that there is one item per agent (numbers of each are the same).
            assert(agents.size() == items.size());

            // Sort agents by income and items by quality (increasing), so that the pairing has the highest items allocated to the highest
            // earners and vice versa.
            std::sort(agents.begin(), agents.end(), [](A a, A b) { return a.income() < b.income(); });
            std::sort(items.begin(), items.end(), [](I a, I b) { return a.quality() < b.quality(); });

            allocations.reserve(items.size());
            // Combine the items and agents into one for convenience for the solver.
            for (size_t i = 0; i < items.size(); i++) {
                // Set the initial guess price to an arbitrary guess according to the function p_i = guess_factor * y_i
                const float guess_price = guess_factor * agents[i].income();
                Allocation<A, I> allocation = {
                    .item = items[i],
                    .agent = agents[i],
                    .price = guess_price,
                    .utility = agents[i].utility(guess_price, items[i].quality())
                };

                allocations.push_back(allocation);
            }
        }

        std::vector<Allocation<A, I>> get_allocations() {
            return std::vector<Allocation<A, I>>(this->allocations);
        }

        /**
         * Solves the allocation model to achieve Pareto efficiency among agents.
         * The algorithm assigns items to agents in a way that no agent can be made better off
         * without making another agent worse off. It iteratively adjusts allocations and prices,
         * possibly swapping agents to improve overall efficiency.
         *
         * The function visualizes the progress by drawing on the specified SFML window.
         *
         * @param epsilon The tolerance for numerical approximations (default is 1e-5).
         * @return A reference to the vector of allocations after solving.
         */
        std::vector<Allocation<A, I>>& solve(RenderState<A, I>* render_state, float epsilon = 1e-5) {
            // If there are no agents, return the empty allocations vector.
            if (allocations.empty()) {
                return allocations;
            }

            // Initialize the first allocation:
            // Assign the first (lowest income) agent to the lowest quality item at zero price.
            // This sets a baseline, as the lowest agent cannot pay less than zero.
            allocations[0].set_price(0.0f);

            // Iterate through each agent starting from the second one.
            // The goal is to assign items to agents in a way that is efficient and respects their preferences.
            for (size_t i = 1; i < allocations.size(); ++i) {
                // Initialize 'agent_to_displace' to -1, indicating no displacement needed initially.
                ssize_t agent_to_displace = -1;

                // References to the current allocation 'a' and the previous allocation 'l'.
                Allocation<A, I>& a = allocations[i];     // Current allocation

                // Determine the 'efficient_price' at which a previous agent 'k' is indifferent
                // between their own allocation and the current allocation 'a'.
                // TODO: issue with efficient price being chosen above agent's income (I think)... what to do???
                float efficient_price;
                size_t k;             // Index of the agent whose indifference sets the price.
                size_t next_k = i - 1; // Start with the previous agent.

                // Loop to find the correct 'k' where earlier agents do not prefer the current allocation.
                do {
                    k = next_k;
                    Allocation<A, I>& indiff = allocations[k];

                    // The maximum price is limited by the agent's income minus epsilon.
                    float max_price = indiff.agent.income() - epsilon;

                    // Find the price that makes agent 'indiff' indifferent between their own allocation
                    // and the current allocation 'a'. This uses a numerical method of bisection.
                    efficient_price = indifferent_price(indiff.agent, a.quality(), indiff.utility,
                                                        indiff.price, max_price, epsilon);

                    // Handle cases where no exact solution is found.
                    if (std::isnan(efficient_price)) {
                        // Efficient price must be at one of the boundaries (indiff.price or max_price).
                        const float min_boundary_diff = indiff.agent.utility(indiff.price, a.quality()) - indiff.utility;
                        const float max_boundary_diff = indiff.agent.utility(max_price, a.quality()) - indiff.utility;

                        // Choose the boundary that is closest to achieving indifference.
                        if (std::abs(min_boundary_diff) < std::abs(max_boundary_diff))
                            efficient_price = indiff.price;
                        else
                            efficient_price = max_price;
                    }

                    // Check if any earlier agents (from index 0 to k-1) prefer the current allocation at 'efficient_price'.
                    for (ssize_t j = k - 1; j >= 0; --j) {
                        Allocation<A, I>& prev = allocations[j];

                        // Ensure that the 'efficient_price' is within the acceptable range for agent 'prev'.
                        if (efficient_price + epsilon < prev.agent.income() && efficient_price + epsilon > prev.price) {
                            assert(a.quality() >= prev.quality()); // Quality should be non-decreasing.

                            // If agent 'prev' prefers the current allocation at 'efficient_price' over their own allocation.
                            if (prev.agent.utility(efficient_price + epsilon, a.quality()) > prev.utility) {
                                // Update 'next_k' to 'j' to consider this agent in the next iteration.
                                next_k = j;
                                break; // Exit the inner loop to update 'k'.
                            }
                        }
                    }
                    // Repeat the loop if 'next_k' has been updated to an earlier agent.
                } while (next_k < k);

                if (efficient_price > a.agent.income()) {
                    // This price cannot be paid by the agent, thus this allocation does not work.
                    // Given that any price below will be preferred by the agent below, we will need to swap.
                    // We can displace by -1 and see what happens.
                    agent_to_displace = i-1;
                } else {
                    // Calculate the utility of the current agent 'a' at the 'efficient_price'.
                    float efficient_utility = a.agent.utility(efficient_price, a.quality());

                    // Check if the current agent 'a' prefers any of the previous allocations over their own at 'efficient_price'.
                    // If so, mark the agent to displace.
                    float u_max = efficient_utility;
                    for (ssize_t j = i - 1; j >= 0; --j) {
                        const Allocation<A, I>& prev = allocations[j];
                        float u_prev = a.agent.utility(prev.price + 100.f * epsilon, prev.quality());
                        if (u_prev > u_max) {
                            // The current agent 'a' prefers 'prev''s allocation; mark 'prev' as the agent to displace.
                            u_max = u_prev;
                            agent_to_displace = j;
                        }
                    }

                    // If no displacement is needed, update the current allocation's price and utility.
                    if (agent_to_displace == -1) {
                        // Set the price to the 'efficient_price' and update utility for the current allocation.
                        a.price = efficient_price;
                        a.utility = efficient_utility;
                    }
                }

                if (agent_to_displace >= 0) {
                    assert(agent_to_displace < i); // The agent to displace should be at a lower index.

                    // Displace the current agent 'a' to position 'agent_to_displace', shifting other agents accordingly.
                    displace(i, agent_to_displace);

                    // After displacement, we need to revisit allocations to ensure efficiency.
                    // Adjust 'i' to continue checking from the appropriate position.
                    if (agent_to_displace == 0) {
                        // If displaced to the first position, reset 'i' to 0 to start over.
                        i = 0;
                    } else {
                        // Set 'i' to 'agent_to_displace - 1' because the for-loop will increment 'i' next.
                        i = agent_to_displace - 1;
                    }
                }

                if (render_state != nullptr) {
                    if (!render_state->draw_allocations(this->allocations, i)) {
                        return allocations;
                    }
                }
            }
            return allocations;
        }

        bool draw(RenderState<A, I>* render_state) {
            return render_state->draw_allocations(this->allocations, -1);
        }

        void regress_price_on_quality() {
            // Ensure there are enough data points
            if (allocations.size() < 2) {
                std::cerr << "Not enough data points to perform regression." << std::endl;
                return;
            }

            size_t n = allocations.size();
            float sum_x = 0.0f;   // Sum of qualities
            float sum_y = 0.0f;   // Sum of prices
            float sum_xx = 0.0f;  // Sum of qualities squared
            float sum_xy = 0.0f;  // Sum of quality * price

            for (const auto& alloc : allocations) {
                float x = alloc.quality();
                float y = alloc.price;
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }

            float x_bar = sum_x / n;
            float y_bar = sum_y / n;

            float Sxy = sum_xy - n * x_bar * y_bar;
            float Sxx = sum_xx - n * x_bar * x_bar;

            if (Sxx == 0.0f) {
                std::cerr << "Cannot compute regression coefficients; division by zero." << std::endl;
                return;
            }

            float b = Sxy / Sxx;          // Slope
            float a = y_bar - b * x_bar;  // Intercept

            std::cout << "Regression result: price = " << a << " + " << b << " * quality" << std::endl;

            // Optionally, calculate the coefficient of determination (R^2)
            float ss_tot = 0.0f;
            float ss_res = 0.0f;
            for (const auto& alloc : allocations) {
                float x = alloc.quality();
                float y = alloc.price;
                float y_pred = a + b * x;
                ss_tot += (y - y_bar) * (y - y_bar);
                ss_res += (y - y_pred) * (y - y_pred);
            }

            float r_squared = 1 - (ss_res / ss_tot);
            std::cout << "Coefficient of determination (R^2): " << r_squared << std::endl;
        }
    };
}

#endif //SOLVER_H
