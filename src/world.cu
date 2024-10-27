//
// Created by ncbmk on 10/18/24.
//

#include "world.h"

peris::Solver<Household, House> World::solver(double guess_factor) {
    // Perform deep copy on households and houses vectors.
    return peris::Solver<Household, House>(std::vector<Household>(this->households), std::vector<House>(this->houses), guess_factor);
}

bool World::validate() {
    for (auto& a : this->households) {
        if (a.aspiration < 0.0 || a.aspiration > 1.0 || a.inc <= 0.0)
            return false;
    }
    for (auto& s : this->schools) {
        if (s.quality <= 0.0)
            return false;
    }

    return true;
}
