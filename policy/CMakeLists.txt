# find_package(Torch REQUIRED PATHS ~/libtorch)
# find_package(Torch REQUIRED PATHS ~/libtorch2/libtorch)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(policy SHARED
    "policy.cpp" 
    "policy.h"
)

target_link_libraries(policy "${TORCH_LIBRARIES}")

# set_property(TARGET policy PROPERTY CXX_STANDARD 17)
