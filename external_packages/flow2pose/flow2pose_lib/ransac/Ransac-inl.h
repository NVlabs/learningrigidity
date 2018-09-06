/*
 * Ransac-inl.h
 * @brief General RANSAC algorithm
 * @date Feb 25, 2010
 * @author Manohar Paluri
 * @author Frank Dellaert
 * @author Chris Beall
 */

#pragma once

#include "Ransac.h"

#include <ctime>
#include <cmath>
#include <iostream>

namespace Ransac {

  /* *************************************************************************** */
  template<class Datum, class Model>
  struct Ransac {

    Parameters parameters_;
    const size_t nPoints;          ///< number of putatives
    size_t nGoodPoints;            ///< number of putatives flagged for use
    std::vector<bool> bestSetMask; ///< best inlier mask
    std::vector<size_t> putmap;    ///<
    size_t maxIter; // maximum number of iterations will be changed adaptively
    size_t bestSetCount;           ///< best number of inliers
    double bestSetError;           ///< error of best estimate
    boost::optional<Model> bestModel_; ///< the best model determined so far
    size_t bestIteration;          ///< Iteration with best Model

    /* ************************************************************************* */
    /** Constructor of everything we need */
    Ransac(size_t setSize, const std::vector<Datum> &ps, Mask &inliers,
        const Parameters& parameters) :
      parameters_(parameters), nPoints(ps.size()), nGoodPoints(0), bestSetMask(
          nPoints), putmap(nPoints), maxIter(INT_MAX), bestSetCount(0),
          bestSetError(std::numeric_limits<double>::max()),
          bestIteration(0) {

      if (parameters_.verbose >= LEVEL0) std::cout << "STARTED RANSAC with "
          << ps.size() << " putatives\n";

      // Make a map of the pre-set mask of inliers, for choosing random
      // putatives, and find the actual number of good points
      if (inliers.size()) {
        for (size_t i = 0; i < nPoints; i++)
          if (inliers[i] == true) {
            putmap[nGoodPoints] = i;
            nGoodPoints++;
          }
      } else {
        // all data will be used for choosing the random sample set
        for (size_t i = 0; i < nPoints; i++)
          putmap[i] = i;
        nGoodPoints = nPoints;
      }

      if (nGoodPoints >= setSize) {
        if (parameters_.verbose >= LEVEL0)
          std::cout << "number of good putatives: " << nGoodPoints << std::endl;
      }
      else if (parameters_.verbose >= LEVEL1)
        std::cout << "RANSAC: Only "<< nGoodPoints << " putatives, aborting.\n";
    }

    /* ************************************************************************* */
    std::vector<bool> sampleSetMask(size_t setSize, const std::vector<Datum> &ps) {
      std::vector<size_t> set(setSize);
      std::vector<bool> setMask(ps.size());

      // Choose a random set, w/out duplicates
      for (size_t i = 0; i < setSize; i++) {
#ifndef _MSC_VER
        retry: set[i] = putmap[::random() % nGoodPoints];
#else
        retry: set[i] = putmap[::rand() % nGoodPoints];
#endif
        for (size_t j = 0; j < i; j++) {
          if (set[j] == set[i]) goto retry;
          // Make sure the coordinates are also unique, otherwise
          // the problem will be underdetermined
          // TODO - not general                if(p1[set[j]].x == p2[set[i]].x && p1[set[j]].y == p2[set[i]].y)
          //                    goto retry;
        }
        //pset.push_back(ps[set[i]]);
        setMask[set[i]] = true;
      }
      if (parameters_.verbose >= LEVEL2) {
        std::sort(set.begin(), set.end());
        for (size_t i = 0; i < setSize; i++)
          std::cout << set[i] << "\t";
        std::cout << std::endl;
      }
      return setMask;
    }

    /* ************************************************************************* */
    void updateMaxIter(size_t setCount, size_t iter, size_t setSize) {
      // Update the best set if the one found is the best;
      // for msac, use robust error to compare, otherwise the number of inliers

      if (setCount == nGoodPoints) {
        maxIter = iter + 1;
        if (parameters_.verbose >= LEVEL2) {
          std::cout << "setCount = nGoodPoints" << std::endl;
          std::cout << "RANSAC: Changed number of iterations to " << maxIter << std::endl;
        }
      } else {
        // Using proportion of inliers found, reduce maximum iterations
        // if necessary
        double newMax = (log(1.0 - double(parameters_.confidence)) / log(1.0
            - pow(double(setCount) / double(nGoodPoints), double(setSize))));
        if (std::isfinite(newMax) && newMax < double(maxIter)) {
          maxIter = (size_t) newMax;
          if (parameters_.verbose >= LEVEL1) {
            std::cout << "RANSAC: Changed number of iterations to " << maxIter
                << std::endl;
          }
        }
      }
    }

    /* ************************************************************************* */
    /** Do a single round */
    bool loop(size_t iter, size_t setSize, boost::function<boost::optional<Model>(const std::vector<
        Datum> &, const Mask &, boost::optional<Model> )> rskernel, boost::function<size_t(const std::vector<
        Datum> &, const Model &, double, Mask &, double *)> rsinliers,
        const std::vector<Datum> &ps, Mask &inliers) {
      if (parameters_.maxIt > 0 && iter > parameters_.maxIt) {
        iter = maxIter;
        if (parameters_.verbose >= LEVEL1) std::cout
            << "RANSAC: maxIt reached, stopping";
        return false;
      }

      if (parameters_.verbose >= LEVEL1) if (iter % 2000 == 0) std::cout
          << "RANSAC: Iteration " << iter << std::endl;

      /// Create a sample set Mask to use for the kernel function
      std::vector<bool> kernelMask = sampleSetMask(setSize, ps);

      // Compute model, go to next iteration immediately if computation fails
      boost::optional<Model> minimalModel = rskernel(ps, kernelMask, boost::none);
      if (!minimalModel) return true;

      // Determine inliers
      std::vector<bool> setMask(nPoints);

      if (inliers.size() == 0)
        fill(setMask.begin(), setMask.end(), true); // Reset the setMask either to all true...
      else
        copy(inliers.begin(), inliers.end(), setMask.begin());// or to the input mask

      double setError = 0.;
      size_t setCount = rsinliers(ps, *minimalModel, parameters_.sigma,
          setMask, parameters_.msac ? (&setError) : NULL);

      if (((parameters_.msac && setError < bestSetError) || (!parameters_.msac
          && setCount > bestSetCount)) && setCount >= setSize) {
        if (parameters_.verbose >= LEVEL1) {
          std::cout << "RANSAC: Set of " << setCount;
          if (parameters_.msac) std::cout << " with error " << setError;
          std::cout << " (iteration " << iter << ")\n";
        }
        bestSetCount = setCount;
        bestSetError = setError;
        bestModel_.reset(*minimalModel);
        bestIteration = iter;
        copy(setMask.begin(), setMask.end(), bestSetMask.begin());
        updateMaxIter(setCount, iter, setSize);
      }

      return true;
    }// Ransac inner loop

    /* ************************************************************************* */
    /** Finish up */
    size_t finish(size_t setSize, boost::function<boost::optional<Model>(const std::vector<Datum> &,
        const Mask &, boost::optional<Model> )> rskernel, //
        boost::function<size_t(const std::vector<Datum> &, const Model &, double,
            Mask &, double *)> rsinliers, //
        const std::vector<Datum> &ps, Model &model, Mask &inliers) {

      // Return failure if not enough points were in any consensus
      if (bestSetCount < setSize) {
        if (inliers.size()) fill(inliers.begin(), inliers.end(), false);
        return 0;
      }

      if (parameters_.verbose >= LEVEL1) {
        std::cout << "RANSAC: " << 100.f * double(bestSetCount)
            / double(nGoodPoints) << "% inliers\n";
      }

      // Compute the final model using all consensus points
      if (parameters_.verbose >= LEVEL1) {
        std::cout << "RANSAC: Computing final matrix\n";
      }
      // Pass bestModel which can be used by kernel function as initialization
      boost::optional<Model> new_model= rskernel(ps, bestSetMask, bestModel_);
      if (new_model)
        model = *new_model;

      if (parameters_.paranoid) {
        // reestimate inliers to refine model;
        // modifies bestSetCount and inliers... not sure if that is a good idea
        if (parameters_.verbose >= LEVEL1) std::cout
            << "RANSAC: Reestimating inliers to obtain a refined model (paranoid flag)\n";

        copy(inliers.begin(), inliers.end(), bestSetMask.begin());

        bestSetCount = rsinliers(ps, model, parameters_.sigma, bestSetMask,
            NULL);

        // Return failure if not enough points were in any consensus
        if (bestSetCount < setSize) {
          if (inliers.size()) fill(inliers.begin(), inliers.end(), false);
          return 0;
        }

        boost::optional<Model> new_model1 = rskernel(ps, bestSetMask, model);
        if (new_model1)
          model = *new_model1;
      }

      // Set the inliers mask if needed
      if (inliers.size()) copy(bestSetMask.begin(), bestSetMask.end(),
          inliers.begin());

      // Return the number of points in agreement with the model
      return bestSetCount;
    }
  };


  /* ************************************************************************* */
  template<class Estimator>
  Result ransac(const std::vector<typename Estimator::Datum> &ps, typename Estimator::Model &model,
      Mask &inliers, const Parameters& parameters) {

    typedef Ransac<typename Estimator::Datum, typename Estimator::Model> MyRansac;
    Result result;

    MyRansac info(Estimator::setSize, ps, inliers, parameters);
    if (info.nGoodPoints < Estimator::setSize) return result;

    size_t iter = 0;
    // Iterate the RANSAC inner loop, selecting a random sample and evaluating it

    while(iter < info.maxIter) {
      if (!info.loop(iter, Estimator::setSize, Estimator::refine, Estimator::inliers, ps, inliers)) break;
      iter++;
    }

    // finish up and prepare result
    result.inlierCount =  info.finish(Estimator::setSize, Estimator::refine, Estimator::inliers, ps, model, inliers);
    result.iterations = iter;
    result.inlierRatio = (double)result.inlierCount/info.nGoodPoints;
    result.bestIteration = info.bestIteration;

    return result;
  }

/* *************************************************************************** */

} // namespace Ransac
