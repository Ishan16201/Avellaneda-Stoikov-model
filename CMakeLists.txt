cmake_minimum_required(VERSION 3.25)
project(project3_integration VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PROJECT3_BUILD_TESTS "Build Project 3 integration tests." ON)
option(PROJECT3_ENABLE_CUDA
  "Build the optional CUDA independent-Hawkes backend (CPU remains the reference)."
  OFF)
set(PROJECT3_LOB_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../Hawkes-driven-limit-order-book-simulator."
    CACHE PATH "Path to the completed Hawkes LOB simulator source tree.")

find_package(Threads REQUIRED)
find_package(Git QUIET)
set(PROJECT3_GIT_REVISION "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE PROJECT3_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

if(NOT EXISTS "${PROJECT3_LOB_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "Hawkes LOB simulator not found at '${PROJECT3_LOB_SOURCE_DIR}'. "
    "Set PROJECT3_LOB_SOURCE_DIR to its source directory.")
endif()

set(LOB_BUILD_TESTS OFF)
set(LOB_BUILD_BENCHMARKS OFF)
add_subdirectory("${PROJECT3_LOB_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/hawkes-lob" EXCLUDE_FROM_ALL)

add_library(project3_integration
  src/avellaneda_stoikov.cpp
  src/batch_hawkes.cpp
  src/benchmark.cpp
  src/differentiable_surrogate.cpp
  src/engine_profile.cpp
  src/exchange_adapter.cpp
  src/execution_layer.cpp
  src/hawkes_policy.cpp
  src/market_quality.cpp
  src/policy_optimization.cpp
  src/reduced_state_solver.cpp
)
target_include_directories(project3_integration PUBLIC include)
target_link_libraries(project3_integration PUBLIC lob::core Threads::Threads)
target_compile_features(project3_integration PUBLIC cxx_std_20)
target_compile_definitions(project3_integration PUBLIC
  PROJECT3_VERSION="${PROJECT_VERSION}"
  PROJECT3_GIT_REVISION="${PROJECT3_GIT_REVISION}"
)

if(PROJECT3_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(project3_cuda STATIC src/cuda/batch_hawkes.cu)
  target_include_directories(project3_cuda PUBLIC include)
  set_target_properties(project3_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_features(project3_cuda PUBLIC cuda_std_17)
  target_link_libraries(project3_integration PUBLIC project3_cuda)
  target_compile_definitions(project3_integration PUBLIC PROJECT3_HAS_CUDA=1)
endif()

if(MSVC)
  target_compile_options(project3_integration PRIVATE /W4 /permissive-)
else()
  target_compile_options(project3_integration PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_executable(project3_demo examples/basic_agent.cpp)
target_link_libraries(project3_demo PRIVATE project3_integration)

add_executable(avellaneda_stoikov_demo examples/avellaneda_stoikov.cpp)
target_link_libraries(avellaneda_stoikov_demo PRIVATE project3_integration)

add_executable(hawkes_policy_comparison examples/hawkes_policy_comparison.cpp)
target_link_libraries(hawkes_policy_comparison PRIVATE project3_integration)

add_executable(reduced_state_convergence examples/reduced_state_convergence.cpp)
target_link_libraries(reduced_state_convergence PRIVATE project3_integration)

add_executable(differentiable_surrogate_demo
  examples/differentiable_surrogate.cpp)
target_link_libraries(differentiable_surrogate_demo PRIVATE project3_integration)

add_executable(policy_optimization_demo
  examples/policy_optimization.cpp)
target_link_libraries(policy_optimization_demo PRIVATE project3_integration)

add_executable(market_quality_stress
  examples/market_quality_stress.cpp)
target_link_libraries(market_quality_stress PRIVATE project3_integration)

add_executable(project3_benchmark
  examples/benchmark_experiment.cpp)
target_link_libraries(project3_benchmark PRIVATE project3_integration)

add_executable(project3_engine_profile
  examples/engine_profile.cpp)
target_link_libraries(project3_engine_profile PRIVATE project3_integration)

if(PROJECT3_BUILD_TESTS)
  enable_testing()
  add_executable(project3_integration_tests tests/integration_tests.cpp)
  target_link_libraries(project3_integration_tests PRIVATE project3_integration)
  add_test(NAME project3_integration_tests COMMAND project3_integration_tests)
endif()
