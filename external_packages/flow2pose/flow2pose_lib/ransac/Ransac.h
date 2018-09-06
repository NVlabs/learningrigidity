/*
 * Ransac.h
 * @brief General RANSAC algorithm
 * @date Feb 25, 2010
 * @author Manohar Paluri
 * @author Frank Dellaert
 * @author Chris Beall
 */

#pragma once

#include "sacCommon.h"
#include <vector>
#include <limits>
#include <iostream>
#include <boost/function.hpp>
#include <boost/optional.hpp>

namespace Ransac {

  /* Exception class */
  class CalibException {
  public:
    const char *msg;
    bool fatal;
    CalibException(const char *msg, bool fatal = 1) {
      std::cout << "CalibException: " << msg << "\n";
    }
    virtual ~CalibException() {
    }
  };

  enum verbosityLevel {
    SILENT, LEVEL0, LEVEL1, LEVEL2
  };

  /*  RANSAC parameters
   *
   * confidence: Chance of a correct result (ie. probability that at least one of the
   * random samples is free from outliers)
   *
   * sigma: Standard deviation for determining inliers. The actual threshold
   * must be derived depending on the application, ie. for problems with
   * codimension 1 the correct value is 3.84*sigma^2, see HZ book for more
   * details TODO: not done !!!!!!
   *
   * verbose: Print RANSAC details during executation.
   *
   * maxIt: maximum number of iterations (for real-time applications); ignored
   * if set to 0 (default)
   *
   * msac: If true, uses likelihood instead of number of inliers to determine
   * the best model. See Torr96cviu (MLESAC: A new robust estimator with
   * application to estimating image geometry). Requires inliers to return a
   * robus error measure by means of optional parameter, see CamMat for an
   * example.
   *
   * paranoid: Reestimate inliers at the end based on the model estimate
   * obtained from all RANSAC inliers
   */
  struct Parameters {
    double confidence, sigma;
    verbosityLevel verbose;
    size_t maxIt;
    bool msac, paranoid;

    Parameters(double _confidence, double _sigma, verbosityLevel _verbose =
        SILENT, size_t _maxIt = 0, bool _msac = false, bool _paranoid = false) :
      confidence(_confidence), sigma(_sigma), verbose(_verbose), maxIt(_maxIt),
          msac(_msac), paranoid(_paranoid) {
      // Input validity checks
      if (confidence <= 0.f || confidence >= 1.f) throw CalibException(
          "RANSAC: 0 < confidence < 1 violated");
    }
  };

  typedef std::vector<bool> Mask;

  /* General RANSAC algorithm
   *
   * See FundMat.{h,cpp} for additional documentation and examples
   *
   * setSize: Number of putatives to try at a time, and pass to the
   * kernel function.  Note that more than this will be passed to the
   * kernel function after the last iteration, to compute the final
   * result, if mat is not NULL.
   *
   * rskernel, rsinliers: Kernel and inlier selection functions.  Note
   * that both functions must obey an initial mask that is passed in.
   * See the description of the inliers parameter.
   *
   * ps: Putatives
   *
   * mat: Pointer to the matrix to hold the result, as computed by
   * rskernel.  If this is not NULL, rskernel will be called with all
   * the inliers after the last iteration to compute the final result.
   *
   * inliers: Array of flags that should have the same number of
   * elements as the putatives, ps.  If inliers is not NULL, it serves
   * two functions.  First, it specifies a mask of putatives that
   * ransac will ignore if their flags in this array are false.
   * Second, any putatives that ransac found to be outliers will be
   * set to false upon successful completion.  Note that the rsinliers
   * function must be written to obey the initial mask.
   *
   * Note: if ransac returns 0 (could not find a good solution), 1
   * (not enough putatives) or throws an exception, none of the output
   * parameters will be modified.
   *
   * Returns the number of inliers on success.  Returns 0 if a good matrix was
   * not found.  Returns 1 if there were not enough putatives to form a
   * set. Throws CalibException on bad input or internal errors.
   * w: (1-epsilon) or the proportion of inliers (worst case! gets adapted automatically)
   * G: (p) or the probability that at least one sample of size setSize will have no outliers
   */

  /**
   * New-style, with templated class
   * The template argument, Estimator, is a class that can estimate a geometric model
   * It needs
   * - a typedef "Model" indicating what the geometric model is
   * - a typedef "Datum" to say what type the putative datums have
   * - a static member sampleSize saying how many datums are needed, minimally
   * - a method "fromMinimal" able to estimate the model from a minimal set of datums // TODO
   *   vector<Datum> -> Model
   * - a method "inliers" to determine whether datums satisfy a model
   *   vector<Datum>  * Model * double * Mask * (double*)-> size_t
   * - a method "refine" which refines the initial estimate
   *   vector<Datum> * Mask -> Model
   */
  template<class Estimator>
  Result ransac(const std::vector<typename Estimator::Datum> &ps,
      typename Estimator::Model &model, Mask &inliers,
      const Parameters& parameters);

} // namespace Ransac
