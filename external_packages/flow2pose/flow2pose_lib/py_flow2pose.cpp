/**
 * Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
 * Licensed under the CC BY-NC-SA 4.0 license 
 * (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
 * 
 * Author: Zhaoyang Lv 
 */

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include "flow2pose.h"

#include "Python.h"
#include "pyboostcvconverter.hpp"

#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Matrix.h>

#include <opencv2/core/core.hpp>

#include <iostream>

using namespace boost::python;
using namespace gtsam;

// transform the vector to a numpy array
static object array_to_nparray(double* data, npy_intp size) {
  npy_intp shape[1] = { size }; // array size
  PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE, // data type
                              NULL, data, // data pointer
                              0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                              NULL);
  handle<> array( obj );
  return object(array);
}

// transform the numpy array to a vector
static std::vector<Point3> nparray_to_vector(PyArrayObject* np_array) {
  // get the size
  npy_intp length = PyArray_SIZE(np_array);
  int size = length / 6;
  // get the pointer to the array
  double* c_array = (double*) PyArray_GETPTR1( np_array, 0 );

  std::vector<Point3> points;
  for (int idx = 0; idx < size; idx ++) {
    Point3 pt (c_array[idx*6], c_array[idx*6 + 1], c_array[idx*6 + 2]);
    points.push_back(pt);
  }
  return points;
}

static Pose3 nparray_to_pose(PyArrayObject* np_array) {
  float* c_array = (float*) PyArray_GETPTR1( np_array, 0 );

  Matrix4 pose_matrix;
  pose_matrix << c_array[0], c_array[1], c_array[2], c_array[3],
   c_array[4], c_array[5], c_array[6], c_array[7],
   c_array[8], c_array[9], c_array[10], c_array[11],
   0, 0, 0, 1;

   return Pose3(pose_matrix);
}

struct pyFlow2Pose : Flow2Pose {

// calculate the transform given the raw vertices map
PyObject* calculate_transform(PyObject *vertices0, PyObject *vertices1, PyObject *flow, PyObject *foreground0, PyObject *foreground1, PyObject *occlusion, PyObject* initial_pose) {
    cv::Mat vMat0, vMat1, flowMat, segMat0, segMat1, occMat;
    vMat0 = pbcvt::fromNDArrayToMat(vertices0);
    vMat1 = pbcvt::fromNDArrayToMat(vertices1);
    flowMat= pbcvt::fromNDArrayToMat(flow);
    segMat0 = pbcvt::fromNDArrayToMat(foreground0);
    segMat1 = pbcvt::fromNDArrayToMat(foreground1);
    occMat = pbcvt::fromNDArrayToMat(occlusion);

    Pose3 pose_init = nparray_to_pose((PyArrayObject*) initial_pose);

    Pose3 result = Flow2Pose::calculate_transform(vMat0, vMat1, flowMat, segMat0, segMat1, occMat, pose_init);

    return gtsam_pose3_to_pyobject(result);
}

// calculate the transform given the matched correspondences
PyObject* solve_pose(PyObject* py_pts1, PyObject* py_pts2, PyObject* py_pose) {
  PyArrayObject* np_pts1 = (PyArrayObject*) py_pts1;
  PyArrayObject* np_pts2 = (PyArrayObject*) py_pts2;
  PyArrayObject* np_pose = (PyArrayObject*) py_pose;
  double* pose_ptr = (double*) PyArray_GETPTR1( np_pose, 0 );

  const std::vector<Point3> pts1 = nparray_to_vector(np_pts1);
  const std::vector<Point3> pts2 = nparray_to_vector(np_pts2);

  Pose3 initial_pose(Rot3::Rodrigues(pose_ptr[0], pose_ptr[1], pose_ptr[2]), Point3(pose_ptr[3], pose_ptr[4], pose_ptr[5]));

  Pose3 result = Flow2Pose::solve(pts1, pts2, initial_pose);

  return gtsam_pose3_to_pyobject(result);
}

PyObject* gtsam_pose3_to_pyobject(const gtsam::Pose3& pose) {
  Matrix4 pose_out = pose.matrix();
  // transform the estimated pose into an numpy object

  double pose_array[16];
  for (auto u = 0; u < 4; u++) {
    for (auto v = 0; v < 4; v++) {
      pose_array[u*4 +v] = pose_out(u, v);
    }
  }

  npy_intp dims[2]{4,4};
  PyObject* np_pose_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA((PyArrayObject*)np_pose_out), pose_array, sizeof(pose_array));
  return np_pose_out;
}

};

void test(PyObject* obj) {

  PyArrayObject* np_obj = (PyArrayObject*) obj;

  npy_intp s = PyArray_SIZE(np_obj);
  // get the pointer to the array
  double* c_array = (double*) PyArray_GETPTR1( np_obj, 0 );

  std::cout << "the size of the array is :" << s << std::endl;

  for (int idx=0; idx < s; idx++) {
    std::cout << c_array[idx] << std::endl;
  }

  std::cout << "Just to show it passed test" << std::endl;
}

BOOST_PYTHON_MODULE(pyFlow2Pose) {

  // numpy requires this
  import_array();

  def("test", &test);

  class_<pyFlow2Pose>("pyFlow2Pose")
    .def( "calculate_transform", &pyFlow2Pose::calculate_transform )
    .def( "solve_pose", &pyFlow2Pose::solve_pose )
    .def_readwrite("solver_type", &pyFlow2Pose::solver_type)
    .def_readwrite("huber_threshold", &pyFlow2Pose::huber_threshold);
}
