/*
 * sacCommon.h
 * @brief Common result return structure for RANSAC/PROSAC algorithms
 * @date Feb 26, 2014
 * @author Chris Beall
 */

#include <iostream>
#include <string>

#pragma once

namespace Ransac {

  struct Result {
    size_t inlierCount;      ///< Number of inliers found by RANSAC/PROSAC
    double inlierRatio;      ///< Inlier ratio
    size_t iterations;       ///< Number of iterations
    size_t bestIteration;    ///< Iteration with max inlierCount, or lowest error

    Result() : inlierCount(0), inlierRatio(0), iterations(0),
        bestIteration(0) {}

    /// print results
    void print(const std::string& s = "") const {
      std::cout << s;
      std::cout << "Inlier count: " << inlierCount << std::endl <<
          "Inlier Ratio: " << inlierRatio << std::endl <<
          "Iterations: " << iterations << std::endl <<
          "Best Iteration: " << bestIteration << std::endl;
    }
  };

} // namespace Ransac
