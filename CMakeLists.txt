cmake_minimum_required(VERSION 3.0.0)
project(DSA_ImageCompressionproject VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 98)
add_executable(DSA_ImageCompressionproject main.cpp compress.cpp)

target_link_libraries(DSA_ImageCompressionproject ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
