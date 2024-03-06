cmake_minimum_required(VERSION 3.12)

project(libtorch_test_ubuntu)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 可执行文件
add_executable(main "main.cpp" "policy/policy.cpp")

add_subdirectory(policy)

target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 17)