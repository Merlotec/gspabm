//
// Created by ncbmk on 10/16/24.
//

#ifndef AGENT_H
#define AGENT_H

#include "solver.h"
#include <vector>

#ifndef __ssize_t_defined
typedef long int ssize_t;
#endif

struct Household {
    float inc;
    float ability;
    float aspiration;

    // Endogenous variables
    ssize_t school;
    ssize_t house;

    float contribution;

    /// The income for the agent.
    float income() const {
        return inc;
    }

    /// This is the utility function used for this agent.
    float utility(float price, float quality) const {
        // Additively separable utility function.
        return powf((inc - price), (1 - aspiration)) + powf(10.f * quality, aspiration);
    }

    std::string debug_info() const {
        return "y=" + std::to_string(inc) + "\nas=" + std::to_string(aspiration) + "\nab=" + std::to_string(ability);
    }
};

struct School {
    ssize_t capacity; // Capacity of the school.
    float x;
    float y;
    float quality;

    // Endogenous variables
    float attainment;
    int num_pupils; // Will be different from size if school is not full.
};

struct House {
    float x;
    float y;

    /// The (best) school allocated to this house.
    ssize_t school; // -1 means no school, -2 means invalid house.

    /// Store quality here to improve cache locality and avoid having to lookup quality from school.
    /// MUST remember to update this every time the school is changed.
    float school_quality;

    __device__ __host__ bool is_valid() const {
        return school >= -1;
    }

    __device__ __host__ bool has_school() const {
        return school >= 0;
    }

    __device__ void invalidate() {
        school = -2;
    }

    float quality() const {
        return school_quality;
    }

    void set_school(ssize_t school, float quality) {
        this->school = school;
        this->school_quality = quality;
    }
};

class World {
    std::vector<Household> households;
    std::vector<School> schools;
    std::vector<House> houses;

public:
    World(std::vector<Household> households, std::vector<School> schools, std::vector<House> houses) : households(households), schools(schools), houses(houses) {}

    /// Returns a solver object from the data in this world. This is used to solve for the optimal prices.
    /// Note: this copies the data from the world instead of moving it.
    /// Maybe could switch to move semantics?
    peris::Solver<Household, House> solver(float guess_factor = 0.2f);

    /// Updates the world to reflect the solution from the solver.
    void reflect_solution(peris::Allocation<Household, House> solution_allocation);

    /// Checks that the households and schools have valid values (such that the preferences will be well-behaved).
    /// Essentially we check that aspiration is between 0 and 1, educational quality is positive, and income is positive for all agents.
    bool validate();
};

#endif //AGENT_H
