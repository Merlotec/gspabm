//
// Created by ncbmk on 10/17/24.
//

#ifndef PERIS_H
#define PERIS_H

#include <string>

#ifndef __ssize_t_defined
typedef long int ssize_t;
#endif

/// Pareto efficient relative investment solver (PERIS).
namespace peris {
    template<typename A>
    concept AgentConcept = requires(A agent, float price, float quality)
    {
        { agent.income() } -> std::same_as<float>;
        { agent.utility(price, quality) } -> std::same_as<float>;
        { agent.debug_info() } -> std::same_as<std::string>;
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
        I item;

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
}

#endif // PERIS_H
