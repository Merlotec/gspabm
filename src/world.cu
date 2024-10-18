//
// Created by ncbmk on 10/18/24.
//

#include "world.h"

peris::Solver<Household, House> World::solver(float guess_factor) {
    // Perform deep copy on households and houses vectors.
    return peris::Solver<Household, House>(std::vector<Household>(this->households), std::vector<House>(this->houses), guess_factor);
}
