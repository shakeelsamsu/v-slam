cmake_minimum_required(VERSION 3.10)

project(BundleAdjustmentTest)

add_executable(bundle_adjustment_test bundle_adjustment_test.cpp)

find_package(Ceres REQUIRED)

target_link_libraries(bundle_adjustment_test Ceres::ceres)