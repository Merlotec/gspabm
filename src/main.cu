#include <iostream>
#include <cuda.h>
#include "world.h"
#include "peris.h"
#include <random>
#include <algorithm>
#include <chrono>
#include <assert.h>

World create_world(const size_t school_count, const size_t house_count) {

    // Number of school places must be exact, so school_count divide into house_count perfectly.
    assert(house_count % school_count == 0);

    const ssize_t school_capacity = house_count / school_count;

    // Initialize with capacities
    std::vector<School> schools;
    schools.reserve(school_count);

    std::vector<House> houses;
    houses.reserve(house_count);

    std::vector<Household> households;
    households.reserve(house_count);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> school_quality_distribution(0.8, 0.25);
    std::normal_distribution<double> ability_distribution(0, 1.0);
    std::normal_distribution<double> aspiration_distribution(0.6, 0.15);

    constexpr double mean_household_inc = 120.0;

    double cv = 0.3; // Adjusts variance and skewness of distribution
    double variance = (cv * mean_household_inc) * (cv * mean_household_inc);
    double standard_deviation = std::sqrt(variance);

    // Calculate the parameters m and s for the underlying normal distribution
    double sigma_squared = std::log((variance / (mean_household_inc * mean_household_inc)) + 1.0);
    double sigma = std::sqrt(sigma_squared);
    double mu = std::log(mean_household_inc) - (sigma_squared / 2.0);

    // Construct the lognormal distribution with parameters mu and sigma
    std::lognormal_distribution<double> household_income_distribution(mu, sigma);

    std::uniform_real_distribution<double> location_axis_distribution(-1.0, 1.0);

    std::cout << "Created distributions" << std::endl;
    for (int i = 0; i < school_count; ++i) {
        // Sample
        const double quality = school_quality_distribution(gen);
        const double x = location_axis_distribution(gen);
        const double y = location_axis_distribution(gen);

        const School school = {.capacity = school_capacity, .x = x, .y = y, .quality = quality, .attainment = -1.f, .num_pupils = 0 };

        schools.push_back(school);
    }

    std::sort(schools.begin(), schools.end(), [](const School& a, const School& b) {
        return a.quality < b.quality;
    });

    std::cout << "Created schools" << std::endl;

    for (int i = 0; i < house_count; ++i) {
        // Sample
        const double x = location_axis_distribution(gen);
        const double y = location_axis_distribution(gen);

        const House house = {.x = x, .y = y, .school = -1 };
        houses.push_back(house);
    }

    std::cout << "Created houses" << std::endl;

    std::vector<House> allocated_houses;
    allocated_houses.reserve(houses.size());

    for (size_t sc = 0; sc < schools.size(); ++sc) {
        School& school = schools[sc];
        std::sort(houses.begin(), houses.end(), [school](const House& a, const House& b) {
            const double a_dis = (a.x - school.x) * (a.x - school.x) + (a.y - school.y) * (a.y - school.y);
            const double b_dis = (b.x - school.x) * (b.x - school.x) + (b.y - school.y) * (b.y - school.y);

            return a_dis > b_dis;
        });

        size_t n = std::min((size_t)school.capacity, houses.size());
        for (size_t i = 0; i < n; ++i) {
            House h = houses.back();
            h.set_school(sc, schools[sc].quality);
            allocated_houses.push_back(h);
            houses.pop_back();
        }
    }
    std::cout << "Allocated houses: " << allocated_houses.size() << std::endl;
    assert(allocated_houses.size() == house_count);



    for (int i = 0; i < house_count; ++i) {
        const double income = household_income_distribution(gen);
        const double ability = ability_distribution(gen);
        const double aspiration = aspiration_distribution(gen);

        const Household household = {.id = i, .inc = income, .ability = ability, .aspiration = aspiration, .school = -1, .house = -1 };
        households.push_back(household);
    }

    std::cout << "Created households: " << households.size() << std::endl;

    return World(households, schools, allocated_houses);
}

__global__ void calculate_contribution(Household* households, int num_households) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_households) {
        Household& hh = households[idx];
        hh.contribution = hh.inc * hh.ability * hh.aspiration;
    }
}

__global__ void sum_contributions(Household* households, int household_count, School* schools, int school_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < household_count) {
        Household& hh = households[idx];

        ssize_t school_idx = hh.school;
        if (school_idx >= 0 && school_idx < school_count) {
            atomicAdd(&(schools[school_idx].attainment), hh.contribution);
            atomicAdd(&(schools[school_idx].num_pupils), 1);
        }
    }
}

__device__ __host__ double household_utility(const Household* hh, double educ, double bid) {
    constexpr double alpha = 0.1;
    const double utility = powf(hh->inc - bid, (1 - alpha)) * powf(educ, alpha);
    return utility;
}

__device__ __host__ double solve_bid(const Household* hh, double e) {
    const double U0 = household_utility(hh, 1.0f, 0.0f);

    double tol = 1e-5f;
    int max_iter = 100;
    double a = 0.0f;
    double b = hh->inc;
    double c, Uc;

    double Ub = household_utility(hh, e, b);
    if (Ub > U0) {
        return b;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        c = 0.5f * (a + b);
        Uc = household_utility(hh, e, c);

        if (fabsf(Uc - U0) < tol) {
            return c;
        }

        if (Uc > U0) {
            a = c;
        } else {
            b = c;
        }
    }
    return c;
}

__global__ void determine_bids(Household* households, int household_count, School* schools, int school_count, House* houses, int house_count, double* valuation_matrix) {
    int hh_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int house_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (hh_idx < household_count && house_idx < house_count) {
        const Household* hh = &households[hh_idx];
        const School& school = schools[houses[hh->house].school];
        const double avg_attainment = school.attainment / school.num_pupils;
        const double bid = solve_bid(hh, avg_attainment);

        valuation_matrix[house_idx * house_count + hh_idx] = bid;
    }
}

int main() {
    constexpr size_t school_count = 100;
    constexpr size_t house_count = 100;

    std::cout << "Creating with " << school_count << " schools" << " and " << house_count << " houses" << std::endl;

    World world = create_world(school_count, house_count);

    std::cout << "Created world" << std::endl;

    assert(world.validate());

    auto solver = world.solver();

    // Setup render window to draw visuals.
    peris::RenderState<Household, House> render_state(solver.get_allocations());

    solver.draw(&render_state);

    auto pres = solver.solve(&render_state);

    std::cout << "Solver finished with code " << pres << std::endl;

    if (solver.verify_solution(0.001)) {
        std::cout << "Verification successful" << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }

    // Run regression to calculate gsp:

    solver.regress_price_on_quality();

    while (solver.draw(&render_state)) {}
    return 0;
}
