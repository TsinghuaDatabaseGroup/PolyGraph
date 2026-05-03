
cmake_minimum_required(VERSION 3.10)
project(PolyGraph)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) 指定默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 2) 对 Release 配置高优化和本机指令集
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

add_definitions(-DEIGEN_NO_DEBUG)

# 包含头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件
file(GLOB SOURCES 
    ${PROJECT_SOURCE_DIR}/src/naive/*.cpp
    ${PROJECT_SOURCE_DIR}/src/construct/*.cpp
    ${PROJECT_SOURCE_DIR}/src/search/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(main ${SOURCES} ${PROJECT_SOURCE_DIR}/test/main.cpp)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP support")
endif()

# Boost
set(BOOST_ROOT "$ENV{CONDA_PREFIX}")
set(BOOST_INCLUDEDIR "$ENV{CONDA_PREFIX}/include")
set(BOOST_LIBRARYDIR "$ENV{CONDA_PREFIX}/lib")
find_package(Boost 1.55 REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Boost 1.55+ not found")
endif()

# 打印源文件
message(STATUS "Source files:")
foreach(src ${SOURCES})
    message(STATUS "  ${src}")
endforeach()
