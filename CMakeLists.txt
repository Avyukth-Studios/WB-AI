cmake_minimum_required(VERSION 3.16)
project(WB_AI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(wbai SHARED src/WB.cpp Wrapper/Wrapper.cpp)
 target_include_directories(wbai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(wbai_demo examples/demo.cpp src/WB.cpp)
target_include_directories(wbai_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
