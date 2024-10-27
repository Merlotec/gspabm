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
    enum SolutionResult : int {
        err_cycle = -4,
        err_budget_constraint = -3,
        err_nan = -2,
        err_unknown = -1,
        terminated = 0,
        success = 1,
        repeat = 2,
    };

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

        void displace_down(size_t a, size_t b) {
            assert(b > a); // Ensure that the source index 'a' is greater than the destination index 'b'.

            // Temporarily store the agent at position 'b' as it will be overridden.
            auto free_agent = allocations[b].agent;

            // Move the agent from position 'a' to position 'b'.
            allocations[b].agent = allocations[a].agent;
            allocations[b].recalculate_utility(); // Update utility after changing the agent.

            // Shift agents from position 'b+1' to 'a' up by one position.
            // This loop moves each agent into the position of the previous agent.
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

    public:
        Solver(std::vector<A> agents, std::vector<I> items, double guess_factor) {
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
        SolutionResult solve(RenderState<A, I>* render_state, double epsilon = 1e-5, int max_iter = 200) {
            // If there are no agents, return the empty allocations vector.
            if (allocations.empty()) {
                return SolutionResult::success;
            }

            //Perform initial alignment.
            SolutionResult res;
            if ((res = align(render_state, 0, epsilon, max_iter)) < 0) {
                // Exit command.
                return res;
            }

            SolutionResult pres;
            while ((pres = push(epsilon, max_iter)) == SolutionResult::repeat) {
                if (render_state != nullptr) {
                    if (!render_state->draw_allocations(this->allocations, -1)) {
                        return SolutionResult::terminated;
                    }
                }
            }

            if (pres != SolutionResult::success) {
                return pres;
            }

            return SolutionResult::success;
        }

    private:
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

            for (size_t j = 0; j < limit; ++j) {
                if (i != j) {
                    const double u_alt = allocations[i].agent.utility(allocations[j].price + 10.f * epsilon, allocations[j].item.quality());
                    if (u_alt > u_max || (isnan(u_max) && !isnan(u_alt))) {
                        u_max = u_alt;
                        i_max = j;
                    }
                }
            }

            return i_max;
        }

        SolutionResult align(RenderState<A, I>* render_state, size_t i, double epsilon, int max_iter = 100) {
            // Initialize the first allocation:
            // Assign the first (lowest income) agent to the lowest quality item at zero price.
            // This sets a baseline, as the lowest agent cannot pay less than zero.
            if (i == 0) {
                allocations[0].set_price(0.0f);
                i = 1;
            }

            size_t head = i;

            size_t reserve = 0;

            bool in_reserve = false;

            // Iterate through each agent starting from the second one.
            // The goal is to assign items to agents in a way that is efficient and respects their preferences.
            for (;i < allocations.size(); ++i) {
                if (i >= allocations.size() - reserve) {
                    in_reserve = true;
                }
                // Initialize 'agent_to_displace' to -1, indicating no displacement needed initially.
                ssize_t agent_to_displace = -1;
                ssize_t agent_to_promote = -1;

                // References to the current allocation 'a' and the previous allocation 'l'.
                Allocation<A, I>& a = allocations[i];     // Current allocation

                // Determine the 'efficient_price' at which a previous agent 'k' is indifferent
                // between their own allocation and the current allocation 'a'.
                Allocation<A, I>& l = allocations[i - 1];

                // The maximum price is limited by the agent's income minus epsilon.
                double max_price = l.agent.income() - epsilon;

                // Find the price that makes agent 'l' indifferent between their own allocation
                // and the current allocation 'a'. This uses a numerical method of bisection.
                double min_price = l.price == 0 ? epsilon : l.price - epsilon;
                double efficient_price = indifferent_price(l.agent, a.quality(), l.utility,
                                                    min_price, max_price, epsilon, max_iter);

                if (isnan(efficient_price)) {
                    if (l.agent.utility(max_price, a.quality()) > l.utility) {
                        efficient_price = max_price;
                    } else {
                        return SolutionResult::err_nan;
                    }
                }

                if (efficient_price + epsilon > a.agent.income()) {
                    // Find best place to move agent to.
                    size_t new_i = most_preferred(i, false, epsilon); // Do not search above because not yet allocated.
                    agent_to_displace = new_i;
                }

                if (agent_to_displace == -1) {

                    // Check if this agent prefers any previous allocations.
                    double u_max = a.agent.utility(efficient_price - 100.f * epsilon, a.quality());

                    if (isnan(u_max))
                        return SolutionResult::err_nan;

                    for (ssize_t j = i - 1; j >= 0; --j) {
                        const Allocation<A, I>& prev = allocations[j];

                        if (a.agent.income() > prev.price + epsilon) {
                            double u_prev = a.agent.utility(prev.price + 100.f * epsilon, prev.quality());
                            if (isnan(u_prev))
                                return SolutionResult::err_nan;
                            if (u_prev > u_max) {
                                // The current agent 'a' prefers 'prev''s allocation; mark 'prev' as the agent to displace.
                                u_max = u_prev;
                                agent_to_displace = j;
                            }
                        }

                        // Check if this other agent prefers this allocation.
                        // Do not care about previous agent because we expect their utility to be the same due to the indifference condition.
                        if (prev.agent.income() > efficient_price) {
                            if (j < i - 1 && prev.agent.utility(efficient_price + epsilon, a.quality()) > prev.utility) {
                                // Shift that agent to here and recalculate.
                                // We will have to invalidate previous allocations due to this.
                                agent_to_promote = j;
                                // Update the price to reflect this.
                                double max_price = prev.agent.income() - epsilon;
                                double min_price = prev.price == 0 ? epsilon : prev.price - epsilon;
                                double p_doublecross = indifferent_price(prev.agent, a.quality(), prev.utility,
                                                    min_price, max_price, epsilon, max_iter);

                                if (p_doublecross > efficient_price) {
                                    efficient_price = p_doublecross;
                                }
                                break;
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
                            // Find the existing agent that gives this agent highest utility and switch.
                            // Calculate the utility of the current agent 'a' at the 'efficient_price'.
                            double efficient_utility = a.agent.utility(efficient_price, a.quality());
                            if (isnan(efficient_utility))
                                return SolutionResult::err_nan;

                            // Set the price to the 'efficient_price' and update utility for the current allocation.
                            a.price = efficient_price;
                            a.utility = efficient_utility;
                        }



                    }
                }

                if (agent_to_displace >= 0) {
                    // Only promote if we also need to displace, otherwise there is no need.
                    if (agent_to_promote >= 0) {
                        // Blacklist old allocation
                        //allocations[agent_to_promote].blacklist_id = allocations[agent_to_promote].agent.item_id();
                        displace_down(agent_to_promote, allocations.size() - 1);
                        ++reserve;
                        i = max(agent_to_promote - 1L, 0L);
                    } else {
                        assert(agent_to_displace < i); // The agent to displace should be at a lower index.

                        // Displace the current agent 'a' to position 'agent_to_displace', shifting other agents accordingly.
                        displace_up(i, agent_to_displace);
                        i = max(agent_to_displace - 1L, 0L);
                    }
                }

                if (render_state != nullptr) {
                    if (i > head) {
                        if (!render_state->draw_allocations(this->allocations, i)) {
                            return SolutionResult::terminated;
                        }
                    }
                }
                head = max(i, head);
            }
            return SolutionResult::success;
        }

        /// Returns true if any agent was pushed, else false.
        SolutionResult push(double epsilon, int max_iter) {
            size_t updated = 0;
            // Start from top, go backwards and ensure we are not inside above indifference curve.
            for (ssize_t i = allocations.size() - 1; i >= 0; --i) {
                // References to the current allocation 'a' and the previous allocation 'l'.
                Allocation<A, I>& a = allocations[i];     // Current allocation

                double efficient_price = a.price;
                // Check if any earlier agents (from index 0 to k-1) prefer the current allocation at 'efficient_price'.
                for (ssize_t j = allocations.size() - 1; j >= 0; --j) {
                    if (i == j) {
                        continue;
                    }
                    Allocation<A, I>& other = allocations[j];

                    // Ensure that the 'efficient_price' is within the acceptable range for agent 'prev'.
                    if (efficient_price + epsilon < other.agent.income()) {
                        //assert(a.quality() >= other.quality()); // Quality should be non-decreasing.

                        // If agent 'other' prefers the current allocation at 'efficient_price' over their own allocation.
                        if (other.agent.utility(efficient_price + epsilon, a.quality()) > other.utility) {
                            // Update 'next_k' to 'j' to consider this agent in the next iteration.
                            // The maximum price is limited by the agent's income minus epsilon.
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

                // Update to reflect new efficient price.
                if (efficient_price > a.price + epsilon) {
                    // Update efficient price and utility.
                    if (efficient_price > a.agent.income()) {
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
        /// Verifies that the current allocation is a solution.
        bool verify_solution(const double epsilon = 1e-6) const {
            for (size_t i = 0; i < allocations.size(); ++i) {
                double u = allocations[i].agent.utility(allocations[i].price, allocations[i].item.quality());
                if (u != allocations[i].utility) {
                    std::cout << "Agent " << i << " has utility mismatch!" << std::endl;
                    return false;
                }

                for (size_t j = 0; j < allocations.size(); ++j) {
                    if (i != j) {
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
            double sum_x = 0.0f;   // Sum of qualities
            double sum_y = 0.0f;   // Sum of prices
            double sum_xx = 0.0f;  // Sum of qualities squared
            double sum_xy = 0.0f;  // Sum of quality * price

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

            // Optionally, calculate the coefficient of determination (R^2)
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
