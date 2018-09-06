cmake_minimum_required(VERSION 2.8)

project(Flow2pose)

set(CMAKE_MACOSX_RPATH 1)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(Boost 1.54 COMPONENTS program_options filesystem python REQUIRED)

find_package(OpenCV COMPONENTS core imgproc REQUIRED)
message(STATUS "OpenCV include: " ${OpenCV_INCLUDE_DIRS})

find_package(GTSAM REQUIRED)
message(STATUS "gtsam include: " ${GTSAM_INCLUDE_DIR})

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

if(NOT WIN32)
  FIND_PACKAGE(X11)
  FIND_PACKAGE(Threads REQUIRED)
endif()

if(X11_FOUND)
  link_directories(${X11_LIB_DIRS})
  include_directories(${X11_INCLUDE_DIR})
  SET( SYSTEM_LIBS ${SYSTEM_LIBS} ${X11_LIBRARIES} )
endif()

include_directories(BEFORE ${CMAKE_SOURCE_DIR})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${BOOST_LIBRARY_DIRS})

file(GLOB srcs flow2pose_lib/flow2pose*)
add_library(${PROJECT_NAME} SHARED ${srcs})
target_link_libraries(${PROJECT_NAME} gtsam opencv_core opencv_imgproc)

file(GLOB pyboostcv_srcs flow2pose_lib/pyboost_*)
add_library(pyboostcv_bridge SHARED ${pyboostcv_srcs})
target_link_libraries(pyboostcv_bridge opencv_core ${PYTHON_LIBRARIES} ${BOOST_LIBRARIES})

PYTHON_ADD_MODULE(pyFlow2Pose "flow2pose_lib/py_flow2pose.cpp")
target_link_libraries(pyFlow2Pose pyboostcv_bridge Boost::python ${PYTHON_LIBRARIES} gtsam ${PROJECT_NAME} opencv_core opencv_highgui)

add_executable(estimate_pose "estimate_pose.cpp")
target_link_libraries(estimate_pose
    opencv_core opencv_imgproc
    Boost::program_options
    Boost::filesystem
    ${PROJECT_NAME}
    ${SYSTEM_LIBS})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS estimate_pose DESTINATION bin)
