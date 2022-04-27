cmake_minimum_required(VERSION 2.8)
project( test )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( test test.cpp )
add_executable( feature_matcher feature_matcher.cpp )
target_link_libraries( test ${OpenCV_LIBS} )
target_link_libraries( feature_matcher ${OpenCV_LIBS} )

