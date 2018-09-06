/**
 * Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
 * Licensed under the CC BY-NC-SA 4.0 license 
 * (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
 * 
 * Author: Zhaoyang Lv 
 */

#include <iostream>
#include <algorithm>
#include <assert.h>

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>

#include "flow2pose.h"

using namespace std;
using namespace gtsam;
using namespace gtsam::noiseModel;

typedef std::vector<Putative> Putatives;
typedef boost::shared_ptr<Putatives> sharedPutatives;

template<typename T>
T bilinearInterp(const cv::Mat_<T>& src, const double row, const double col) {
  int lower_row = std::floor(row);
  int upper_row = lower_row < src.rows-1 ? lower_row + 1 : lower_row;
  int lower_col = std::floor(col);
  int upper_col = lower_col < src.cols-1 ? lower_col + 1 : lower_col;

  // suppose it's 0-1 grid
  double r_u = upper_row - row; // row to upper row
  double l_r = row - lower_row; // low row to row
  double c_u = upper_col - col; // col to upper col
  double l_c = col - lower_col; // low col to col

  return src(lower_row, lower_col)*r_u*c_u + src(lower_row, upper_col)*r_u*l_c +
      src(upper_row, lower_col)*l_r*c_u + src(upper_row, upper_col)*l_r*l_c;
}

inline bool within_image(int row, int col, int rows, int cols) {
  if (row < 0 || row >= rows || col < 0 || col >= cols) {
    return false;
  } else {
    return true;
  }
}

Flow2Pose::Flow2Pose() {
  measurement_noise_model = Diagonal::Sigmas(Vector3(1, 1, 1));
  huber_measurement_model = Robust::Create(mEstimator::Huber::Create(huber_threshold, mEstimator::Huber::Scalar), measurement_noise_model);
}

Point3 invdepth (const Point3& pt, OptionalJacobian<3,3> J_p = boost::none) {
  double inv_z = 1 / pt.z();
  double inv_z2 = inv_z * inv_z;

  if (J_p) {
    *J_p << inv_z, 0, -pt.x() * inv_z2,
     0, inv_z, -pt.y() * inv_z2,
     0, 0, -inv_z2;
  }
  return Point3(pt.x()*inv_z, pt.y()*inv_z, inv_z);
}

struct Depth {
  double d;
  int i;

  Depth(double depth, int index) {
    d = depth;
    i = index;
  }
};

struct compairDepth {
  bool operator()(const Depth&a, const Depth& b) {
    return a.d < b.d;
  }
} compair_depth;

gtsam::Pose3 Flow2Pose::calculate_transform(const cv::Mat& v_map0, const cv::Mat& v_map1, const cv::Mat& flow, const cv::Mat& foreground_S0, const cv::Mat& foreground_S1, const cv::Mat& occlusion, const Pose3& pose_init) {
  auto rows = v_map0.rows;
  auto cols = v_map0.cols;

  vector<Point3> vertices_0;
  vector<Point3> vertices_1;

  std::vector<Depth> tmp_depth;
  // sharedPutatives putatives(new vector<Putative>());

  // sparsely sample the flow pairs with pad=4
  for (auto row = 8; row < rows-8; row+=4) {
    for (auto col = 8; col < cols-8; col+=4) {
      // check whether this point is a background point and are not occluded.
      // foreground has value 1; occlusion region has value 0
      if (foreground_S0.at<int>(row, col) > 1e-2) continue;
      if (occlusion.at<int>(row, col) < 1e-2) continue;

      // check whether the mapped points are within the image
      const auto& f = flow.at<cv::Vec2f>(row, col);
      auto row_f = round(row + f[1]);
      auto col_f = round(col + f[0]);
      if (!within_image(row_f, col_f, rows, cols)) {
        continue;
      }

      if (bilinearInterp<int>(foreground_S1, row_f, col_f) > 1e-2) {
        // remove the pointst that on the second frame foreground
        continue;
      }

      const auto& v0 = v_map0.at<cv::Vec3d>(row, col);
      const auto& v1 = bilinearInterp<cv::Vec3d>(v_map1, row_f, col_f);
      // const auto& v1 = v_map1.at<cv::Vec3d>(row_f, col_f);

      // truncate points that are too far away to too near the frustum. This can improve the stability.
      if (v0[2] > 5*1e2 || v1[2] > 5*1e2 || v0[2] < 1e-2 || v1[2] < 1e-2) {
        continue;
      }

      tmp_depth.push_back(Depth(v0[2], vertices_0.size()));
      vertices_0.push_back(Point3(v0[0], v0[1], v0[2]));
      vertices_1.push_back(Point3(v1[0], v1[1], v1[2]));
    }
  }

  if (tmp_depth.size() < 10) {
    // too few valida points
    return pose_init;
  }

  // std::sort(tmp_depth.begin(), tmp_depth.end(), compair_depth);

  vector<Point3> v_list0;
  vector<Point3> v_list1;
  // vector<Putative> putatives;
  // Sort and get the top 12000 points near the frustum. It makes the solver much faster.
  double error_threshold = 0.0;
  auto filtered_num = std::min(12000, int(tmp_depth.size()));
  for (auto i = 0; i < filtered_num; i++) {
    int index = tmp_depth[i].i;
    v_list0.push_back(Point3(vertices_0[index]));
    v_list1.push_back(Point3(vertices_1[index]));
    // putatives->push_back(Putative(Point3(vertices_0[index]), Point3(vertices_1[index])));
  }

  vector<bool> inliers(tmp_depth.size(), true);

  error_threshold /= filtered_num;
  error_threshold = v_list0[filtered_num/5].norm();
  error_threshold = min(error_threshold/10, 15.00);
  cout << "the error threshold is: " << error_threshold << endl;

  for (auto i = 0; i < filtered_num; i++) {
    if ((v_list0[i] - v_list1[i]).norm() > error_threshold) {
      inliers[i] = false;
    }
  }

  Pose3 pose(pose_init);

  return solve(v_list0, v_list1, pose, inliers);
}

Pose3 Flow2Pose::solve(const vector<Point3>& pts1, const vector<Point3>& pts2, const Pose3& initial_pose, const vector<bool>& inliers) {
    int total_num = pts1.size();
    assert(total_num == pts2.size());

    Symbol var_pose('p', 0);
    Pose3_ pose(var_pose);

    ExpressionFactorGraph graph;
    // evaluate all the points first

    // add all factors from the list
    for(int idx = 0; idx < total_num; idx++) {
        if (inliers.size() > 0) {
          if(!inliers[idx]) continue;
        }

        const Point3& pt1 = pts1[idx];
        const Point3& pt2 = pts2[idx];
        // set up the factor graph for solving the camera pose
        Point3_ pt2_(pt2);
        // point 2 that is transformed to point 1
        Point3_ pt2_1_ = transform_to(pose, pt2_);

        graph.addExpressionFactor(pt2_1_, pt1, huber_measurement_model);
    }

    // specify the uncertainty on the pose
    Values initial;
    initial.insert(var_pose, initial_pose);

    if (solver_type == Gauss_Newton) {

      static GaussNewtonParams parameters;
      parameters.relativeErrorTol = 1e-8;
      // Do not perform more than N iteration steps
      // parameters.maxIterations = 100;
      parameters.verbosity = NonlinearOptimizerParams::ERROR;
      GaussNewtonOptimizer optimizer(graph, initial, parameters);

      try {
        Values result = optimizer.optimize();
        return result.at<Pose3>(var_pose);
      } catch (IndeterminantLinearSystemException& e) {
        cout << e.what() << endl;
        return Pose3();
      }

    } else if (solver_type == Levenberg_Marquardt) {
      static LevenbergMarquardtParams parameters;
      parameters.relativeErrorTol = 1e-8;
      // Do not perform more than N iteration steps
      parameters.maxIterations = 100;
      parameters.verbosity = NonlinearOptimizerParams::ERROR;
      parameters.diagonalDamping = true;
      LevenbergMarquardtOptimizer optimizer(graph, initial, parameters);

      Values result = optimizer.optimize();
      return result.at<Pose3>(var_pose);
    } else {
      cout << "The solver is not implemented." << endl;
      exit(1);
    }

}