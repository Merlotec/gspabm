//
// Created by ncbmk on 10/17/24.
//

#ifndef PERIS_H
#define PERIS_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <cassert>
#include <iostream>

/// Pareto efficient relative investment solver (PERIS).
namespace peris {
    template<typename A>
    concept AgentConcept = requires(A agent, float price, float quality)
    {
        { agent.income() } -> std::same_as<float>;
        { agent.utility(price, quality) } -> std::same_as<float>;
    };

    template<typename I>
    concept ItemConcept = requires(I item)
    {
        { item.quality() } -> std::same_as<float>;
    };

    /// Represents a single agent in the model, who wishes to maximize utility.
    template<typename A, typename I>
        requires AgentConcept<A> && ItemConcept<I>
    struct Allocation {
        /// The item being allocated.
        const I item;

        /// The agent being allocated to this item. Note - this can change, hence is not const.
        A agent;

        /// The current allocation price.
        float price;

        /// The current allocation utility for the agent.
        float utility;

        float quality() const {
            return item.quality();
        }

        void set_price(float price) {
            this->price = price;
            recalculate_utility();
        }

        void recalculate_utility() {
            utility = agent.utility(price, quality());
        }
    };

    template<typename A>
        requires AgentConcept<A>
    float indifferent_quality(A &agent, float price, float u_0, float y_min, float y_max, float epsilon = 1e-4,
                              int max_iter = 100) {
        float lower = y_min;
        float upper = y_max;
        float mid = 0.0f;
        int iter = 0;

        while (iter < max_iter) {
            mid = (lower + upper) / 2.0f;
            float u_mid = agent.utility(price, mid);
            float diff = u_mid - u_0;

            if (std::abs(diff) < epsilon)
                return mid;

            if (diff > 0)
                upper = mid;
            else
                lower = mid;

            iter++;
        }

        // Return NaN (not a number) if solution was not found within the tolerance of epsilon
        return std::numeric_limits<float>::quiet_NaN();
    }

    template<typename A>
        requires AgentConcept<A>
    float indifferent_price(A &agent, float quality, float u_0, float x_min, float x_max, float epsilon = 1e-4,
                            int max_iter = 100) {
        float lower = x_min;
        float upper = x_max;
        float mid = 0.0f;
        int iter = 0;
        float u_mid;
        while (iter < max_iter) {
            mid = (lower + upper) / 2.0f;
            u_mid = agent.utility(mid, quality);
            float diff = u_mid - u_0;

            if (std::abs(diff) < epsilon)
                return mid;

            // Because the utility function is increasing in quality we swap this from the solver for quality.
            if (diff > 0)
                lower = mid;
            else
                upper = mid;

            iter++;
        }

        // Return NaN if solution was not found within tolerance
        return std::numeric_limits<float>::quiet_NaN();
    }


    template<typename A, typename I>
        requires AgentConcept<A> && ItemConcept<I>
    bool draw_allocations(sf::RenderWindow &window, std::vector<peris::Allocation<A, I> > &allocations) {
        float x_min_base = allocations.front().price * 0.9;
        float x_max_base = allocations.back().price * 1.1;

        float y_min_base = allocations.front().item.quality();
        float y_max_base = allocations.back().item.quality() * 1.1;

        // Add padding
        float x_padding = abs(x_max_base - x_min_base) * 0.05f;
        float y_padding = abs(y_max_base - y_min_base) * 0.05f;

        float x_min = x_min_base - x_padding;
        float y_min = y_min_base - y_padding;

        float x_max = x_max_base + x_padding;
        float y_max = y_max_base +  y_padding;

        const float x_scale = window.getSize().x / (x_max - x_min);
        const float y_scale = window.getSize().y / (y_max - y_min);

        // Process events
        sf::Event event{};
        while (window.pollEvent(event)) {
            // Close window if requested
            if (event.type == sf::Event::Closed) {
                window.close();
                return false;
            }
        }

        // Clear the window with a white background
        window.clear(sf::Color::White);

        // **Draw axes**
        {
            sf::VertexArray axes(sf::Lines);

            // X-axis (horizontal line at y = 0 or at y_min if y = 0 is not in range)
            float x_axis_y;
            if (y_min <= 0 && y_max >= 0) {
                // y = 0 is within y range
                x_axis_y = window.getSize().y - (-y_min) * y_scale;
            } else {
                // y = 0 is not within range; draw x-axis at y_min
                x_axis_y = window.getSize().y - (0 - y_min) * y_scale;
            }

            // Draw the x-axis line
            axes.append(sf::Vertex(sf::Vector2f(0, x_axis_y), sf::Color::Black));
            axes.append(sf::Vertex(sf::Vector2f(window.getSize().x, x_axis_y), sf::Color::Black));

            // Y-axis (vertical line at x = 0 or at x_min if x = 0 is not in range)
            float y_axis_x;
            if (x_min <= 0 && x_max >= 0) {
                // x = 0 is within x range
                y_axis_x = (-x_min) * x_scale;
            } else {
                // x = 0 is not within range; draw y-axis at x_min
                y_axis_x = (0 - x_min) * x_scale;
            }

            // Draw the y-axis line
            axes.append(sf::Vertex(sf::Vector2f(y_axis_x, 0), sf::Color::Black));
            axes.append(sf::Vertex(sf::Vector2f(y_axis_x, window.getSize().y), sf::Color::Black));

            // Draw the axes on the window
            window.draw(axes);
        }

        // Draw indifference curves for each utility level
        for (const auto &a: allocations) {
            //std::cout << "Allocation: p:" << a.price << ", e:" << a.quality() << ", u:" << a.utility << std::endl;
            sf::VertexArray curve(sf::LineStrip);

            // Sample points along x-axis to plot the curve
            for (float x = x_min_base; x <= x_max_base; x += 0.05f) {
                // Find y such that U(x, y) = U0
                float y = indifferent_quality(a.agent, x, a.utility, y_min, y_max);

                // Check if y is valid
                if (!std::isnan(y)) {
                    float screen_x = (x - x_min) * x_scale;
                    float screen_y = window.getSize().y - (y - y_min) * y_scale;
                    curve.append(sf::Vertex(sf::Vector2f(screen_x, screen_y), sf::Color::Red));
                }
            }
            window.draw(curve);

            float screen_x = (a.price - x_min) * x_scale;
            float screen_y = window.getSize().y - (a.quality() - y_min) * y_scale;

            sf::CircleShape circle(5); // Circle with radius 5 pixels
            circle.setFillColor(sf::Color::Blue);
            circle.setPosition(screen_x - 5, screen_y - 5); // Center the circle
            window.draw(circle);
        }

        // Display the current frame
        window.display();
        sf::sleep(sf::microseconds(100));

        return true;
    }

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

        // Insert agent a into agent b's position, and shift everything up one.
        void displace(size_t a, size_t b) {
            assert(b < a);

            // Put overridden agent into free agent buffer to be dealt with.
            auto free_agent = allocations[b].agent;

            allocations[b].agent = allocations[a].agent;
            allocations[b].recalculate_utility();

            for (size_t i = b + 1; i <= a; ++i) {
                auto agent_buffer = allocations[i].agent;
                allocations[i].agent = free_agent;
                allocations[i].recalculate_utility();
                free_agent = agent_buffer;
            }
            // Last agent is agent a which has already been moved to b so we don't need to worry about allocating this free agent.
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

        /// Solves the model to a precision of `epsilon`. Draws the progress graphically on the specified window.
        std::vector<Allocation<A, I> > &solve_visual(sf::RenderWindow &window, float epsilon = 1e-5) {
            // If there are no agents then they are vacuously solved.
            if (allocations.empty()) {
                return allocations;
            }
            // We start by assuming the first agent will get the lowest quality item, since due to the relative nature
            // of this problem, it necessary for the lowest agent to pay zero (otherwise they could all improve utility
            // by shifting down, until the lowest agent pays zero, which is the constraint we impose to get absolute
            // numbers).
            allocations[0].set_price(0.f);

            // Iterate through the rest of the allocations (from 1 to n). Note: we may need to backtrack to the first agent if the second (or any subsequent agent) should be placed in allocation 0.
            // TODO: there is a slight worry about a cycle taking place where two allocations keep swapping, if they have very close utility functions, and the error margin of the computational solution results in contradicting comparisons.
            for (size_t i = 1; i < allocations.size(); ++i) {
                // We allocate one agent at a time, iteratively. First, we must check if this agent prefers the (previous) lower agent's allocation (which is designed to be worse for them).
                // If it is greater, we want to swap this agent with the lower agent. Once we swap, we must continue with this process until it is no longer necessary to swap (so we must check for n-2 as well, and if there is a swap, n-3 etc. etc.)
                // After a swap, we need to recalculate the allocations for all the newly displaced agents.
                ssize_t agent_to_displace = -1;

                Allocation<A, I> &a = allocations[i];
                Allocation<A, I> &l = allocations[i - 1];

                // If this condition holds, this agent prefers the lower allocation, and the lower agent prefers this allocation, so they should swap.
                // In theory this could be removed and the code should still work, but this allows us to avoid the costly operation of calculating if we can clearly see that they should swap.
                if (a.agent.utility(l.price + epsilon, l.quality()) > a.utility && l.agent.utility(
                        a.price + epsilon, a.quality()) > l.utility) {
                    agent_to_displace = i - 1;
                }

                if (agent_to_displace == -1) {
                    // Now we want to find the price that makes the lower agent (i-1) indifferent between choosing their allocation and this new allocation.
                    float efficient_price;
                    // Now see if any agents i - 1 or below prefer this allocation to their own.
                    size_t k = i - 1;
                    while (true) {
                        Allocation<A, I> &indiff = allocations[k];
                        // Assuming well-behaved preferences, the price must be weakly greater than the current price and less than the agent's income.
                        float max_price = indiff.agent.income() - epsilon;

                        // Function may break down at zero consumption, so we choose a value epsilon away).
                        efficient_price = indifferent_price(indiff.agent, a.quality(), indiff.utility, indiff.price, max_price,
                                                                       epsilon);

                        // If there is no solution, it is probably because the solution is on the edges of the specified solution range.
                        if (isnan(efficient_price)) {
                            // Efficient price must be at boundaries.
                            const float min_boundary = indiff.agent.utility(indiff.price, a.quality()) - indiff.utility;
                            const float max_boundary = indiff.agent.utility(max_price, a.quality()) - indiff.utility;
                            // Check which boundary is closer to the solution.
                            if (abs(min_boundary) < abs(max_boundary))
                                efficient_price = indiff.price;
                            else
                                efficient_price = max_price;
                        }

                        for (ssize_t j = k - 1; j >= 0; --j) {
                            Allocation<A, I> &prev = allocations[j];
                            // If this other previously allocated agent prefers this allocation over their own then we have a problem.
                            // We instead want to use this agent as the one who's indifference sets the price of the current allocation i.
                            if (efficient_price + epsilon < prev.agent.income() && efficient_price + epsilon > prev.price) {
                                assert(a.quality() >= prev.quality());
                                if (prev.agent.utility(efficient_price + epsilon, a.quality()) > prev.utility) {
                                    // Update the indifferent agent to this new agent (which will iterate the while loop again).
                                    k = j;
                                    goto continue_while; // Continues outer loop.
                                }
                            }
                        }
                        // No previous agents prefer another allocation, so we can proceed.
                        break;
                        continue_while:;
                    }

                    // Calculate utility associated with this new 'efficiently' priced allocation.
                    float efficient_utility = a.agent.utility(efficient_price, a.quality());

                    // If this agent prefers the lower allocation at this 'efficient' price we need to swap.
                    // Include epsilon here to stop cycles (due to issues in approximating with many agents) since epsilon is the 'worst case' scenario, and if this inequality holds with epsilon then we definitely need to swap.
                    // If not, then the difference is small enough that the utilities are practically identical.
                    for (ssize_t j = i - 1; j >= 0; --j) {
                        Allocation<A, I> &prev = allocations[j];
                        const float u_prev = a.agent.utility(prev.price + epsilon, prev.quality());
                        if (u_prev > efficient_utility) {
                            // We prefer agent j's allocation, so would like to switch.
                            agent_to_displace = j;
                        }
                    }

                    // If we still do not want to swap agents then assign the efficient price. Everything up until now is efficient.
                    if (agent_to_displace == -1) {
                        // Set the price to the efficient price corresponding to this agent - it doesn't matter if this agent ends up being swapped.
                        a.price = efficient_price;
                        // Rather than calling set price we do it manually since we have already calculated the utility.
                        a.utility = efficient_utility;
                    }
                }

                if (agent_to_displace >= 0) {
                    assert(agent_to_displace < i); // Should only swap down.

                    // Swap agents.
                    displace(i, agent_to_displace);

                    // Because we have moved the current agent to the previous agent, but we now need to check if it should be moved back even further or ensure that the price is efficient for that new agent.
                    if (agent_to_displace == 0) {
                        // Will hit zero so we cannot push this agent further back, or adjust the price of the allocation 0, so we want to only recalculate the agent now allocated to 1.
                        // This will keep i the same for the next iteration but will operate on a different agent because of the swap.
                        i = 0;
                    } else {
                        // This means that the next iteration will be i = i-1 (since it is incremented by 1 by the for loop).
                        i = agent_to_displace - 1;
                    }
                }

                // Draw the current state of the world to see progress.
                if (!draw(window)) {
                    return allocations;
                }
            }
            return allocations;
        }

        bool draw(sf::RenderWindow &window) {
            return draw_allocations(window, this->allocations);
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

#endif // PERIS_H
