cmake_minimum_required(VERSION 3.20)
project(WB_AI LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

Python3_add_library(_wb_ai MODULE WITH_SOABI cpp/python_module.cpp)
target_link_libraries(_wb_ai PRIVATE llama)
target_include_directories(_wb_ai PRIVATE ${Python3_INCLUDE_DIRS})
set_target_properties(_wb_ai PROPERTIES PREFIX "")
