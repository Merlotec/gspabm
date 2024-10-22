//
// Created by ncbmk on 10/17/24.
//

#ifndef PERIS_H
#define PERIS_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <iomanip>

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
        float x_min_base = 0.f;//allocations.front().price * 0.9;
        float x_max_base = allocations.back().price * 1.1;

        float y_min_base = 0.f;//allocations.front().item.quality();
        float y_max_base = allocations.back().item.quality() * 1.1;

        // Add padding
        float x_padding = abs(x_max_base - x_min_base) * 0.1f;
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

        sf::Font font;
        if (!font.loadFromFile("Arial.ttf")) {
            // Handle error
            std::cerr << "Error loading font!" << std::endl;
            return false;
        }

        // Render axes after to control z-order.
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

            // **Draw axis labels**
            // X-axis label
            sf::Text x_label("Price", font, 14);
            x_label.setFillColor(sf::Color::Black);
            x_label.setPosition(window.getSize().x - 50, x_axis_y - 20); // Adjust position as needed
            window.draw(x_label);

            // Y-axis label
            sf::Text y_label("Quality", font, 14);
            y_label.setFillColor(sf::Color::Black);
            y_label.setPosition(y_axis_x + 5, 5); // Adjust position as needed

            window.draw(y_label);

             // **Draw tick marks and labels with regular intervals**

            // Helper function to calculate 'nice' numbers for intervals
            auto calculate_nice_number = [](float range, bool round) {
                float exponent = std::floor(std::log10(range));
                float fraction = range / std::pow(10, exponent);

                float nice_fraction;
                if (round) {
                    if (fraction < 1.5f)
                        nice_fraction = 1.0f;
                    else if (fraction < 3.0f)
                        nice_fraction = 2.0f;
                    else if (fraction < 7.0f)
                        nice_fraction = 5.0f;
                    else
                        nice_fraction = 10.0f;
                } else {
                    if (fraction <= 1.0f)
                        nice_fraction = 1.0f;
                    else if (fraction <= 2.0f)
                        nice_fraction = 2.0f;
                    else if (fraction <= 5.0f)
                        nice_fraction = 5.0f;
                    else
                        nice_fraction = 10.0f;
                }

                return nice_fraction * std::pow(10, exponent);
            };

            // X-axis ticks
            int desired_x_ticks = 5;
            float x_range = x_max - x_min;
            float x_tick_interval = calculate_nice_number(x_range / (desired_x_ticks - 1), true);
            float x_nice_min = std::floor(x_min / x_tick_interval) * x_tick_interval;
            float x_nice_max = std::ceil(x_max / x_tick_interval) * x_tick_interval;

            for (float x_value = x_nice_min; x_value <= x_nice_max; x_value += x_tick_interval) {
                float screen_x = (x_value - x_min) * x_scale;

                // Draw tick
                sf::Vertex tick[] = {
                    sf::Vertex(sf::Vector2f(screen_x, x_axis_y - 5), sf::Color::Black),
                    sf::Vertex(sf::Vector2f(screen_x, x_axis_y + 5), sf::Color::Black)
                };
                window.draw(tick, 2, sf::Lines);

                // Draw label
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << x_value;
                sf::Text label(ss.str(), font, 12);
                label.setFillColor(sf::Color::Black);
                label.setPosition(screen_x - 15, x_axis_y + 10); // Adjust position as needed
                window.draw(label);
            }

            // Y-axis ticks
            int desired_y_ticks = 5;
            float y_range = y_max - y_min;
            float y_tick_interval = calculate_nice_number(y_range / (desired_y_ticks - 1), true);
            float y_nice_min = std::floor(y_min / y_tick_interval) * y_tick_interval;
            float y_nice_max = std::ceil(y_max / y_tick_interval) * y_tick_interval;

            for (float y_value = y_nice_min; y_value <= y_nice_max; y_value += y_tick_interval) {
                float screen_y = window.getSize().y - (y_value - y_min) * y_scale;

                // Draw tick
                sf::Vertex tick[] = {
                    sf::Vertex(sf::Vector2f(y_axis_x - 5, screen_y), sf::Color::Black),
                    sf::Vertex(sf::Vector2f(y_axis_x + 5, screen_y), sf::Color::Black)
                };
                window.draw(tick, 2, sf::Lines);

                // Draw label
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << y_value;
                sf::Text label(ss.str(), font, 12);
                label.setFillColor(sf::Color::Black);
                label.setPosition(y_axis_x - 50, screen_y - 10); // Adjust position as needed
                window.draw(label);
            }
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

        /**
         * Solves the allocation model to achieve Pareto efficiency among agents.
         * The algorithm assigns items to agents in a way that no agent can be made better off
         * without making another agent worse off. It iteratively adjusts allocations and prices,
         * possibly swapping agents to improve overall efficiency.
         *
         * The function visualizes the progress by drawing on the specified SFML window.
         *
         * @param window The SFML window where the allocation progress is visualized.
         * @param epsilon The tolerance for numerical approximations (default is 1e-5).
         * @return A reference to the vector of allocations after solving.
         */
        std::vector<Allocation<A, I>>& solve_visual(sf::RenderWindow& window, float epsilon = 1e-5) {
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

                // Calculate the utility of the current agent 'a' at the 'efficient_price'.
                float efficient_utility = a.agent.utility(efficient_price, a.quality());

                // Check if the current agent 'a' prefers any of the previous allocations over their own at 'efficient_price'.
                // If so, mark the agent to displace.
                for (ssize_t j = i - 1; j >= 0; --j) {
                    const Allocation<A, I>& prev = allocations[j];
                    if (a.agent.utility(prev.price + epsilon, prev.quality()) > efficient_utility) {
                        // The current agent 'a' prefers 'prev''s allocation; mark 'prev' as the agent to displace.
                        agent_to_displace = j;
                    }
                }

                // If no displacement is needed, update the current allocation's price and utility.
                if (agent_to_displace == -1) {
                    // Set the price to the 'efficient_price' and update utility for the current allocation.
                    a.price = efficient_price;
                    a.utility = efficient_utility;
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

                // Draw the current state of allocations on the window for visualization.
                if (!draw(window)) {
                    // If the window is closed or drawing fails, return the current allocations.
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
