#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>

#include <opencv2/core/core.hpp>

#include "ransac/Ransac-inl.h"

struct Putative {
  gtsam::Point3 pt1;
  gtsam::Point3 pt2;

  Putative(const gtsam::Point3 p1, const gtsam::Point3 p2)
  : pt1(p1), pt2(p2) {
  }
};

struct ThreePointRANSAC {
  typedef gtsam::Pose3 Model;          ///< Pose3 is the estimated model
  typedef Putative Datum;
  typedef std::vector<Datum> Datums;

  static const size_t setSize = 3;        ///< minimum sample set

  static bool compute_covariance_;
  static gtsam::SharedNoiseModel sigma_;
  static gtsam::Matrix covariance_;

  static void Set(bool compute_covariance);

  /// RANSAC inlier function
  static size_t inliers(const Datums& putatives, const Model& model,
                        double thresh, Ransac::Mask& mask, double* error);

  /// RANSAC kernel function which computes the model (Pose3)
  static gtsam::Pose3 refine(const Datums& putatives,
      const Ransac::Mask& mask, boost::optional<gtsam::Pose3> bestModel);
};

struct Flow2Pose {

enum {
  Gauss_Newton,
  Levenberg_Marquardt
};

double huber_threshold = 1;
double ransacConfidence =0.999;
double ransacIteration = 1000;
int solver_type = Gauss_Newton;

gtsam::SharedNoiseModel huber_measurement_model;
gtsam::SharedNoiseModel measurement_noise_model;

Flow2Pose();

/* Solve Camera pose */
gtsam::Pose3 solve(const std::vector<gtsam::Point3>& pts1,
  const std::vector<gtsam::Point3>& pts2, const gtsam::Pose3& initial_pose, const std::vector<bool>& inliers = std::vector<bool>());

gtsam::Pose3 calculate_transform(const cv::Mat& v_map0, const cv::Mat& v_map1,
const cv::Mat& flow, const cv::Mat& foreground_S0, const cv::Mat& foreground_S1, const cv::Mat& occlusion, const gtsam::Pose3& initial_pose);

Ransac::Result ransac_solve(const ThreePointRANSAC::Datums& putatives, std::vector<bool>& inliers, gtsam::Pose3& pose, double sigma);

};
