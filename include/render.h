// render.h
#ifndef RENDER_H
#define RENDER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifndef __ssize_t_defined
typedef long int ssize_t;
#endif

// Placeholder for peris::Allocation
namespace peris {

    // Template class RenderState
    template<typename A, typename I>
    class RenderState {
    public:
        // Constructor
        inline RenderState(const std::vector<peris::Allocation<A, I>>& allocations);

        // Update rendering data for changed allocations
        inline void update_allocations(const std::vector<peris::Allocation<A, I>>& new_allocations);

        // Draw allocations
        inline bool draw_allocations(const std::vector<peris::Allocation<A, I>>& new_allocations, int current_idx);

        // Public window field (if needed outside)
        sf::RenderWindow window;

    private:
        int downtime = 1;

        // Fields
        sf::Font font;

        // Precomputed rendering data
        std::vector<sf::VertexArray> indifference_curves;
        std::vector<sf::CircleShape> allocation_circles;

        // Axes and labels (assumed static)
        sf::VertexArray axes;
        std::vector<sf::Text> axis_labels;
        std::vector<sf::VertexArray> ticks;
        std::vector<sf::Text> tick_labels;

        // Coordinate transformations
        float x_min, x_max, y_min, y_max;
        float x_min_base, x_max_base, y_min_base, y_max_base;
        float x_scale, y_scale;
        float x_axis_y, y_axis_x;

        float epsilon = 1e-4;

        // Store previous allocations for comparison
        std::vector<peris::Allocation<A, I>> old_allocations;

        // Helper methods
        inline void compute_coordinate_transformations(const std::vector<peris::Allocation<A, I>>& allocations);
    };

    // Implementation of RenderState methods

    template<typename A, typename I>
    inline RenderState<A, I>::RenderState(const std::vector<peris::Allocation<A, I>>& allocations)
        : old_allocations(allocations), window(sf::VideoMode(1500, 1000), "Pareto Efficient Relative Investment Solver (PERIS)") {
        // Load font
        if (!font.loadFromFile("Arial.ttf")) {
            std::cerr << "Error loading font!" << std::endl;
            // Handle error appropriately
        }

        // Compute coordinate transformations and axes
        compute_coordinate_transformations(allocations);

        // Precompute indifference curves and allocation circles
        for (const auto& a : allocations) {
            // Indifference Curve
            sf::VertexArray curve(sf::LineStrip);

            for (float x = x_min; x <= a.agent.income(); x += 0.05f) {

                float x_use;
                if (x > a.agent.income()) {
                     x_use = a.agent.income() - epsilon;
                } else {
                    x_use = x;
                }

                float y = indifferent_quality(a.agent, x, a.utility, y_min, y_max, epsilon);

                if (!std::isnan(y) && y >= y_min && y <= y_max) {
                    float screen_x = (x - x_min) * x_scale;
                    float screen_y = window.getSize().y - (y - y_min) * y_scale;
                    curve.append(sf::Vertex(sf::Vector2f(screen_x, screen_y), sf::Color::Red));
                }
            }
            indifference_curves.push_back(curve);

            // Allocation Circle
            float screen_x = (a.price - x_min) * x_scale;
            float screen_y = window.getSize().y - (a.quality() - y_min) * y_scale;

            sf::CircleShape circle(5);
            circle.setFillColor(sf::Color::Blue);
            circle.setPosition(screen_x - 5, screen_y - 5);

            allocation_circles.push_back(circle);
        }
    }

    template<typename A, typename I>
    inline void RenderState<A, I>::compute_coordinate_transformations(const std::vector<peris::Allocation<A, I>>& allocations) {
        // Compute bounds based on allocations
        x_min_base = 0.f;
        x_max_base = allocations.back().price * 1.1f;
        y_min_base = 0.f;
        y_max_base = allocations.back().quality() * 1.1f;

        // Add padding
        float x_padding = std::abs(x_max_base - x_min_base) * 0.1f;
        float y_padding = std::abs(y_max_base - y_min_base) * 0.05f;

        x_min = x_min_base - x_padding;
        y_min = y_min_base - y_padding;
        x_max = x_max_base + x_padding;
        y_max = y_max_base + y_padding;

        // Compute scales
        x_scale = window.getSize().x / (x_max - x_min);
        y_scale = window.getSize().y / (y_max - y_min);

        // Precompute axes positions
        if (y_min <= 0 && y_max >= 0) {
            x_axis_y = window.getSize().y - (-y_min) * y_scale;
        } else {
            x_axis_y = window.getSize().y - (0 - y_min) * y_scale;
        }

        if (x_min <= 0 && x_max >= 0) {
            y_axis_x = (-x_min) * x_scale;
        } else {
            y_axis_x = (0 - x_min) * x_scale;
        }

        // Precompute axes
        axes.clear();
        axes.setPrimitiveType(sf::Lines);
        axes.append(sf::Vertex(sf::Vector2f(0, x_axis_y), sf::Color::Black));
        axes.append(sf::Vertex(sf::Vector2f(window.getSize().x, x_axis_y), sf::Color::Black));
        axes.append(sf::Vertex(sf::Vector2f(y_axis_x, 0), sf::Color::Black));
        axes.append(sf::Vertex(sf::Vector2f(y_axis_x, window.getSize().y), sf::Color::Black));

        // Precompute axis labels
        axis_labels.clear();
        sf::Text x_label("Price", font, 14);
        x_label.setFillColor(sf::Color::Black);
        x_label.setPosition(window.getSize().x - 50, x_axis_y - 20);
        axis_labels.push_back(x_label);

        sf::Text y_label("Quality", font, 14);
        y_label.setFillColor(sf::Color::Black);
        y_label.setPosition(y_axis_x + 5, 5);
        axis_labels.push_back(y_label);

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

        // Precompute ticks and labels
        ticks.clear();
        tick_labels.clear();

        // X-axis ticks
        int desired_x_ticks = 5;
        float x_range = x_max - x_min;
        float x_tick_interval = calculate_nice_number(x_range / (desired_x_ticks - 1), true);
        float x_nice_min = std::floor(x_min / x_tick_interval) * x_tick_interval;
        float x_nice_max = std::ceil(x_max / x_tick_interval) * x_tick_interval;

        for (float x_value = x_nice_min; x_value <= x_nice_max; x_value += x_tick_interval) {
            float screen_x = (x_value - x_min) * x_scale;

            // Tick marks
            sf::VertexArray tick(sf::Lines, 2);
            tick[0] = sf::Vertex(sf::Vector2f(screen_x, x_axis_y - 5), sf::Color::Black);
            tick[1] = sf::Vertex(sf::Vector2f(screen_x, x_axis_y + 5), sf::Color::Black);
            ticks.push_back(tick);

            // Labels
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << x_value;
            sf::Text label(ss.str(), font, 12);
            label.setFillColor(sf::Color::Black);
            label.setPosition(screen_x - 15, x_axis_y + 10);
            tick_labels.push_back(label);
        }

        // Y-axis ticks
        int desired_y_ticks = 5;
        float y_range = y_max - y_min;
        float y_tick_interval = calculate_nice_number(y_range / (desired_y_ticks - 1), true);
        float y_nice_min = std::floor(y_min / y_tick_interval) * y_tick_interval;
        float y_nice_max = std::ceil(y_max / y_tick_interval) * y_tick_interval;

        for (float y_value = y_nice_min; y_value <= y_nice_max; y_value += y_tick_interval) {
            float screen_y = window.getSize().y - (y_value - y_min) * y_scale;

            // Tick marks
            sf::VertexArray tick(sf::Lines, 2);
            tick[0] = sf::Vertex(sf::Vector2f(y_axis_x - 5, screen_y), sf::Color::Black);
            tick[1] = sf::Vertex(sf::Vector2f(y_axis_x + 5, screen_y), sf::Color::Black);
            ticks.push_back(tick);

            // Labels
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << y_value;
            sf::Text label(ss.str(), font, 12);
            label.setFillColor(sf::Color::Black);
            label.setPosition(y_axis_x - 50, screen_y - 10);
            tick_labels.push_back(label);
        }
    }

    template<typename A, typename I>
    inline void RenderState<A, I>::update_allocations(const std::vector<peris::Allocation<A, I>>& new_allocations) {
        // Update coordinate transformations
        compute_coordinate_transformations(new_allocations);

        // Ensure the sizes are the same
        if (new_allocations.size() != old_allocations.size()) {
            // If sizes differ, we need to recompute all rendering data
            old_allocations = new_allocations;
            indifference_curves.clear();
            allocation_circles.clear();

            // Recompute all
            for (size_t i = 0; i < new_allocations.size(); ++i) {
                const auto& a = new_allocations[i];

                // Indifference Curve
                sf::VertexArray curve(sf::LineStrip);

                for (float x = x_min; x <= x_max; x += 0.05f) {
                    float x_use;
                    if (x > a.agent.income()) {
                        x_use = a.agent.income() - epsilon;
                    } else {
                        x_use = x;
                    }

                    float y = indifferent_quality(a.agent, x, a.utility, y_min, y_max, epsilon);

                    if (!std::isnan(y) && y >= y_min && y <= y_max) {
                        float screen_x = (x - x_min) * x_scale;
                        float screen_y = window.getSize().y - (y - y_min) * y_scale;
                        curve.append(sf::Vertex(sf::Vector2f(screen_x, screen_y), sf::Color::Red));
                    }
                }
                indifference_curves.push_back(curve);

                // Allocation Circle
                float screen_x = (a.price - x_min) * x_scale;
                float screen_y = window.getSize().y - (a.quality() - y_min) * y_scale;

                sf::CircleShape circle(5);
                circle.setFillColor(sf::Color::Blue);
                circle.setPosition(screen_x - 5, screen_y - 5);

                allocation_circles.push_back(circle);
            }
            return;
        }

        // Compare new allocations with old ones
        for (size_t i = 0; i < new_allocations.size(); ++i) {
            const auto& new_alloc = new_allocations[i];
            const auto& old_alloc = old_allocations[i];

            // Update allocation in old_allocations
            old_allocations[i] = new_alloc;

            // Recompute indifference curve
            sf::VertexArray curve(sf::LineStrip);

            const auto& a = new_alloc;
            for (float x = x_min; x <= x_max; x += 0.05f) {
                float x_use;
                if (x > a.agent.income()) {
                    x_use = a.agent.income() - epsilon;
                } else {
                    x_use = x;
                }

                float y = indifferent_quality(a.agent, x, a.utility, y_min, y_max, epsilon);

                if (!std::isnan(y) && y >= y_min && y <= y_max) {
                    float screen_x = (x - x_min) * x_scale;
                    float screen_y = window.getSize().y - (y - y_min) * y_scale;
                    curve.append(sf::Vertex(sf::Vector2f(screen_x, screen_y), sf::Color::Red));
                }
            }
            indifference_curves[i] = curve;

            // Recompute allocation circle
            float screen_x = (new_alloc.price - x_min) * x_scale;
            float screen_y = window.getSize().y - (new_alloc.quality() - y_min) * y_scale;

            allocation_circles[i].setPosition(screen_x - 5, screen_y - 5);
        }
    }

    template<typename A, typename I>
    inline bool RenderState<A, I>::draw_allocations(const std::vector<peris::Allocation<A, I>>& new_allocations, int current_idx) {
        // Update allocations
        update_allocations(new_allocations);

        // Process events
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                return false;
            }
        }

        // Clear the window
        window.clear(sf::Color::White);

        // Get the mouse position
        sf::Vector2i mouse_position = sf::Mouse::getPosition(window);
        int hovered_allocation_index = -1;

        // Determine if the mouse is over any allocation point
        for (size_t i = 0; i < allocation_circles.size(); ++i) {
            sf::CircleShape& circle = allocation_circles[i];

            // Get the circle's position and radius
            sf::Vector2f position = circle.getPosition();
            float radius = circle.getRadius();

            // Adjust position to center
            position.x += radius;
            position.y += radius;

            // Calculate distance
            float dx = mouse_position.x - position.x;
            float dy = mouse_position.y - position.y;
            float distance = std::sqrt(dx * dx + dy * dy);

            if (distance <= radius) {
                hovered_allocation_index = static_cast<int>(i);
                break;
            }
        }

        // Draw indifference curves
        for (size_t i = 0; i < indifference_curves.size(); ++i) {
            sf::Color curve_color = (static_cast<int>(i) == hovered_allocation_index) ? sf::Color::Cyan : sf::Color::Red;

            if (current_idx >= 0) {
                  if (i > current_idx) {
                      curve_color = sf::Color::Transparent;
                  } else if (i == current_idx) {
                      curve_color = sf::Color::Green;
                  }

            }

            sf::VertexArray& curve = indifference_curves[i];

            // Update color
            for (size_t j = 0; j < curve.getVertexCount(); ++j) {
                curve[j].color = curve_color;
            }
            window.draw(curve);
        }

        // Draw allocation circles
        for (size_t i = 0; i < allocation_circles.size(); ++i) {
            sf::Color circle_color = (static_cast<int>(i) == hovered_allocation_index) ? sf::Color::Cyan : sf::Color::Blue;
            sf::CircleShape& circle = allocation_circles[i];
            circle.setFillColor(circle_color);
            window.draw(circle);
        }

        // Optionally, display agent information when hovered
        if (hovered_allocation_index >= 0) {
            const auto& a = new_allocations[hovered_allocation_index];

            std::ostringstream info;
            info << "p=" << a.price << "\ne=" << a.item.quality() << "\n" << a.agent.debug_info();
            sf::Text agent_info(info.str(), font, 14);
            agent_info.setFillColor(sf::Color::Black);
            agent_info.setPosition(mouse_position.x + 10, mouse_position.y + 10);
            window.draw(agent_info);
        }

        // Draw axes
        window.draw(axes);

        // Draw axis labels
        for (const auto& label : axis_labels) {
            window.draw(label);
        }

        // Draw ticks
        for (const auto& tick : ticks) {
            window.draw(tick);
        }

        // Draw tick labels
        for (const auto& label : tick_labels) {
            window.draw(label);
        }

        // Display the current frame
        window.display();
        sf::sleep(sf::microseconds(downtime));

        return true;
    }
}

#endif // RENDER_H
