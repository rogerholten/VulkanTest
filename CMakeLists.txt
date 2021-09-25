cmake_minimum_required(VERSION 3.20)
project(VulkanTest)

set(CMAKE_CXX_STANDARD 17)

# Vulkan
find_package(Vulkan REQUIRED)

# GLFW
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

include_directories(./lib/glm)

add_executable(VulkanTest src/main.cpp)

target_link_libraries(VulkanTest ${GLFW_STATIC_LIBRARIES} ${Vulkan_LIBRARY})

# execute_process(COMMAND ./spirv.sh)