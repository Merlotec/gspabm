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
	float indifferent_quality(A& agent, float price, float u_0, float y_min, float y_max, float epsilon = 1e-4,
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
	float indifferent_price(A& agent, float quality, float u_0, float x_min, float x_max, float epsilon = 1e-4,
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
		float x_min = allocations.front().price * 0.9;
		float x_max = allocations.back().price * 1.1;

		float y_min = allocations.front().item.quality();
		float y_max = allocations.back().item.quality() * 1.1;

		// Add padding
		float x_padding = abs(x_max - x_min) * 0.01f;
		float y_padding = abs(x_max - x_min) * 0.01f;

		x_min -= x_padding;
		y_min -= y_padding;

		x_max += x_padding;
		y_max += y_padding;

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
			for (float x = x_min; x <= x_max; x += 0.05f) {
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
			allocations[a].recalculate_utility();
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

		/// Solves the model to a precision of `epsilon`. The `shift` parameter specifies the starting offsets to use.
		std::vector<Allocation<A, I> > &solve_visual(sf::RenderWindow &window, float epsilon = 1e-6) {
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
			for (int i = 1; i < allocations.size(); ++i) {
				// We allocate one agent at a time, iteratively. First, we must check if this agent prefers the (previous) lower agent's allocation (which is designed to be worse for them).
				// If it is greater, we want to swap this agent with the lower agent. Once we swap, we must continue with this process until it is no longer necessary to swap (so we must check for n-2 as well, and if there is a swap, n-3 etc. etc.)
				// After a swap, we need to recalculate the allocations for all the newly displaced agents.
				bool should_swap_agents = false;

				Allocation<A, I>& a = allocations[i];
				Allocation<A, I>& l = allocations[i-1];

				// If this condition holds, this agent prefers the lower allocation, and the lower agent prefers this allocation, so they should swap.
				// In theory this could be removed and the code should still work, but this allows us to avoid the costly operation of calculating if we can clearly see that they should swap.
				if (a.agent.utility(l.price + epsilon, l.quality()) > a.utility && l.agent.utility(a.price + epsilon, a.quality()) > l.utility) {
					should_swap_agents = true;
				}

				if (!should_swap_agents) {
					// Now we want to find the price that makes the lower agent (i-1) indifferent between choosing their allocation and this new allocation.
					// Assuming well-behaved preferences, the price must be weakly greater than the current price and less than the agent's income.
					float max_price = l.agent.income() - epsilon; // Function may break down at zero consumption, so we choose a value epsilon away).
					const float efficient_price = indifferent_price(l.agent, a.quality(), l.utility, l.price, max_price, epsilon);

					// Calculate utility associated with this new 'efficiently' priced allocation.
					const float efficient_utility = a.agent.utility(efficient_price, a.quality());

					// If this agent prefers the lower allocation at this 'efficient' price we need to swap.
					// Include epsilon here to stop cycles (due to issues in approximating with many agents) since epsilon is the 'worst case' scenario, and if this inequality holds with epsilon then we definitely need to swap.
					// If not, then the difference is small enough that the utilities are practically identical.
					if (a.agent.utility(l.price + epsilon, l.quality()) > efficient_utility) {
						should_swap_agents = true;
					} else {
						// Set the price to the efficient price corresponding to this agent - it doesn't matter if this agent ends up being swapped.
						a.price = efficient_price; // Rather than calling set price we do it manually since we have already calculated the utility.
						a.utility = efficient_utility;
					}
				}

				if (should_swap_agents) {
					// Swap agents.
					swap_agents(i, i - 1);

					// Because we have moved the current agent to the previous agent, but we now need to check if it should be moved back even further or ensure that the price is efficient for that new agent.
					if (i == 1) {
						// Will hit zero so we cannot push this agent further back, or adjust the price of the allocation 0, so we want to only recalculate the agent now allocated to 1.
						// This will keep i the same for the next iteration but will operate on a different agent because of the swap.
						i -= 1;
					} else {
						// This means that the next iteration will be i = i-1 (since it is incremented by 1 by the for loop).
						i -= 2;
					}
				}

				// Draw the current state of the world to see progress.
				if (!draw(window)) {
					return allocations;
				}
			}
		}

		bool draw(sf::RenderWindow &window) {
			return draw_allocations(window, this->allocations);
		}
	};
}

#endif // PERIS_H
