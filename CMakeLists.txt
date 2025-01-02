cmake_minimum_required(VERSION 3.12)

project(libtorch_test_ubuntu)
set(CMAKE_CXX_STANDARD 17)
if(UNIX)
    add_definitions(-DUNIX)
    find_package(aris REQUIRED PATHS /usr/aris)
endif(UNIX)

include_directories(${aris_INCLUDE_DIRS})

find_package(Torch REQUIRED PATHS /home/kaanh/libtorch/share/cmake/Torch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(policy)

# 可执行文件
add_executable(main "main.cpp")


target_link_libraries(main "${TORCH_LIBRARIES}" policy)

target_link_libraries(main ${aris_LIBRARIES} atomic pthread m)
