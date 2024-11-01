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

    enum RenderCommand : int {
        terminate = 0,
        none = 1,
        pause = 2,
        tick = 3,
        skip = 4,
        enable_swap_always = 5,
    };

    // Template class RenderState
    template<typename A, typename I>
    class RenderState {
    public:
        // Constructor
        inline RenderState();

        // Draw allocations
        inline RenderCommand draw_allocations(const std::vector<peris::Allocation<A, I>>& allocations, int current_idx);

        // Public window field (if needed outside)
        sf::RenderWindow window;

    private:
        // Fields
        sf::Font font;

        // View for zooming and panning
        sf::View view;

        // Zoom level
        float zoom_level = 1.0f;

        // Panning variables
        bool is_panning = false;
        sf::Vector2i last_mouse_position;

        // Data area for initial view setup
        sf::FloatRect data_area;
        bool view_initialized = false;

        // Coordinate bounds
        float x_min, x_max, y_min, y_max;
    };

    // Implementation of RenderState methods

    template<typename A, typename I>
    inline RenderState<A, I>::RenderState()
        : window(sf::VideoMode(1500, 1000), "Pareto Efficient Relative Investment Solver (PERIS)") {
        // Load font
        if (!font.loadFromFile("Arial.ttf")) {
            std::cerr << "Error loading font!" << std::endl;
            // Handle error appropriately
        }
    }

    template<typename A, typename I>
    inline RenderCommand RenderState<A, I>::draw_allocations(const std::vector<peris::Allocation<A, I>>& allocations, int current_idx) {
        // Compute coordinate bounds based on allocations
        if (allocations.empty()) {
            return RenderCommand::none;
        }

        x_min = allocations.front().price;
        x_max = allocations.front().price;
        y_min = allocations.front().quality();
        y_max = allocations.front().quality();

        for (const auto& a : allocations) {
            float x = a.price;
            float y = a.quality();

            if (x < x_min) x_min = x;
            if (x > x_max) x_max = x;
            if (y < y_min) y_min = y;
            if (y > y_max) y_max = y;
        }

        // Add padding
        float x_padding = std::abs(x_max - x_min) * 0.1f;
        float y_padding = std::abs(y_max - y_min) * 0.1f;

        x_min -= x_padding;
        x_max += x_padding;
        y_min -= y_padding;
        y_max += y_padding;

        // Flip y_min and y_max to account for inverted drawing
        std::swap(y_min, y_max);
        y_min = -y_min;
        y_max = -y_max;

        // Set data area for the view
        data_area = sf::FloatRect(x_min, y_min, x_max - x_min, y_max - y_min);

        // Initialize view if not already done
        if (!view_initialized) {
            view.reset(sf::FloatRect(x_min, y_min, x_max - x_min, y_max - y_min));
            view_initialized = true;
        }

        RenderCommand rc = RenderCommand::none;

        // Process events
        sf::Event event{};
        while (window.pollEvent(event)) {
            // Close window if requested
            if (event.type == sf::Event::Closed) {
                window.close();
                return RenderCommand::terminate;
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::T) {
                    rc = RenderCommand::tick;
                } else if (event.key.code == sf::Keyboard::P) {
                    rc = RenderCommand::pause;
                } else if (event.key.code == sf::Keyboard::S) {
                    rc = RenderCommand::skip;
                } else if (event.key.code == sf::Keyboard::E) {
                    rc = RenderCommand::enable_swap_always;
                }
            }

            // Handle zooming with mouse wheel
            if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                    float delta = event.mouseWheelScroll.delta;
                    // Update zoom level
                    if (delta > 0) {
                        zoom_level *= 0.9f; // Zoom in
                    } else if (delta < 0) {
                        zoom_level *= 1.1f; // Zoom out
                    }
                    // Limit zoom level
                    //if (zoom_level < 0.1f) zoom_level = 0.1f;

                    // Update view size
                    sf::Vector2f new_size = sf::Vector2f((x_max - x_min) * zoom_level, (y_max - y_min) * zoom_level);
                    view.setSize(new_size);
                }
            }

            // Handle panning with right mouse button
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Right) {
                    is_panning = true;
                    last_mouse_position = sf::Mouse::getPosition(window);
                }
            }

            if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Right) {
                    is_panning = false;
                }
            }

            if (event.type == sf::Event::MouseMoved) {
                if (is_panning) {
                    sf::Vector2i new_mouse_position = sf::Mouse::getPosition(window);
                    sf::Vector2f old_pos = window.mapPixelToCoords(last_mouse_position, view);
                    sf::Vector2f new_pos = window.mapPixelToCoords(new_mouse_position, view);
                    sf::Vector2f delta = old_pos - new_pos;
                    view.move(delta);
                    last_mouse_position = new_mouse_position;
                }
            }
        }

        // Set the updated view
        window.setView(view);

        // Calculate the scale factors to correct circle scaling
        float scale_x = window.getSize().x / view.getSize().x;
        float scale_y = window.getSize().y / view.getSize().y;
        float aspect_ratio = scale_y / scale_x;

        // Clear the window with a white background
        window.clear(sf::Color::White);

        // Draw indifference curves and allocation circles
        for (size_t i = 0; i < allocations.size(); ++i) {
            const Allocation<A, I>& a = allocations[i];
            sf::VertexArray curve(sf::LineStrip);

            // Sample points along x-axis to plot the curve
            int num_samples = 500;
            for (int j = 0; j <= num_samples; ++j) {
                float x = x_min + j * (x_max - x_min) / num_samples;
                // Find y such that U(x, y) = U0
                float y = indifferent_quality(a.agent, x, a.utility, -y_max, -y_min);

                // Check if y is valid
                if (!std::isnan(y) && y >= -y_max && y <= -y_min) {
                    auto color = (int)i == current_idx ? sf::Color::Green : sf::Color::Red;
                    curve.append(sf::Vertex(sf::Vector2f(x, -y), color));
                }
            }
            window.draw(curve);
        }

        // Draw allocation circles
        for (size_t i = 0; i < allocations.size(); ++i) {
            const Allocation<A, I>& a = allocations[i];

            float x = a.price;
            float y = -a.quality();

            float circle_radius = 0.008f * zoom_level; // Adjust circle size inversely proportional to zoom level
            sf::CircleShape circle(circle_radius);
            auto color = i == current_idx ? sf::Color::Cyan : sf::Color::Blue;
            circle.setFillColor(color);
            circle.setOrigin(circle.getRadius(), circle.getRadius()); // Center the circle
            circle.setPosition(x, y);

            // Adjust the circle's scale to correct aspect ratio
            circle.setScale(aspect_ratio, 1.0f);

            window.draw(circle);
        }

        // Draw axes
        {
            sf::VertexArray axes(sf::Lines);

            // X-axis (horizontal line at y = 0)
            if (-y_min <= 0 && -y_max >= 0) {
                axes.append(sf::Vertex(sf::Vector2f(x_min, 0), sf::Color::Black));
                axes.append(sf::Vertex(sf::Vector2f(x_max, 0), sf::Color::Black));
            }

            // Y-axis (vertical line at x = 0)
            if (x_min <= 0 && x_max >= 0) {
                axes.append(sf::Vertex(sf::Vector2f(0, -y_min), sf::Color::Black));
                axes.append(sf::Vertex(sf::Vector2f(0, -y_max), sf::Color::Black));
            }

            window.draw(axes);

            // Draw axis labels and ticks in screen coordinates
            // Reset the view temporarily
            window.setView(window.getDefaultView());

            // X-axis label
            sf::Text x_label("Price", font, 14);
            x_label.setFillColor(sf::Color::Black);
            x_label.setPosition(window.getSize().x - 70, window.getSize().y - 30); // Adjust position as needed
            window.draw(x_label);

            // Y-axis label
            sf::Text y_label("Quality", font, 14);
            y_label.setFillColor(sf::Color::Black);
            y_label.setPosition(10, 10); // Adjust position as needed
            window.draw(y_label);

            // Draw tick marks and labels
            // X-axis ticks
            int desired_x_ticks = 10;
            float x_range = x_max - x_min;
            float x_tick_interval = x_range / desired_x_ticks;

            for (int i = 0; i <= desired_x_ticks; ++i) {
                float x_value = x_min + i * x_tick_interval;
                sf::Vector2f world_pos(x_value, 0);
                sf::Vector2f screen_pos = sf::Vector2f(window.mapCoordsToPixel(world_pos, view));

                // Draw tick
                sf::RectangleShape tick(sf::Vector2f(1, 5));
                tick.setFillColor(sf::Color::Black);
                tick.setPosition(screen_pos.x, screen_pos.y);
                window.draw(tick);

                // Draw label
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << x_value;
                sf::Text label(ss.str(), font, 12);
                label.setFillColor(sf::Color::Black);
                label.setPosition(screen_pos.x - 15, screen_pos.y + 5);
                window.draw(label);
            }

            // Y-axis ticks
            int desired_y_ticks = 10;
            float y_range = -y_max - (-y_min);
            float y_tick_interval = y_range / desired_y_ticks;

            for (int i = 0; i <= desired_y_ticks; ++i) {
                float y_value = -y_max + i * y_tick_interval;
                sf::Vector2f world_pos(0, y_value);
                sf::Vector2f screen_pos = sf::Vector2f(window.mapCoordsToPixel(world_pos, view));

                // Draw tick
                sf::RectangleShape tick(sf::Vector2f(5, 1));
                tick.setFillColor(sf::Color::Black);
                tick.setPosition(screen_pos.x - 5, screen_pos.y);
                window.draw(tick);

                // Draw label
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << -y_value;
                sf::Text label(ss.str(), font, 12);
                label.setFillColor(sf::Color::Black);
                label.setPosition(screen_pos.x - 50, screen_pos.y - 10);
                window.draw(label);
            }

            // Restore the view
            window.setView(view);
        }

        // Display the current frame
        window.display();

        return rc;
    }

} // namespace peris

#endif // RENDER_H
