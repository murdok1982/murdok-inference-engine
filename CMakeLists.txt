cmake_minimum_required(VERSION 3.20)
project(murdok-inference-engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Optimization flags for MSVC
if (MSVC)
    add_compile_options(/W3 /O2 /Oi)
    # Enable AVX2 on host
    add_compile_options(/arch:AVX2)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O3 -mavx2 -mfma)
endif()

# Upstream llama.cpp submodule options
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama: build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama: build server" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "llama: build tools" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "llama: build common" FORCE)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    add_subdirectory(third_party/llama.cpp)
else()
    message(FATAL_ERROR "third_party/llama.cpp not found. Please run git submodule update --init --recursive")
endif()

# MuRDoK Core Library
set(MURDOK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MURDOK_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(murdok_core STATIC
    ${MURDOK_SRC_DIR}/hardware/hardware_detect.cpp
    ${MURDOK_SRC_DIR}/runtime/engine.cpp
)

target_include_directories(murdok_core PUBLIC
    ${MURDOK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)

target_link_libraries(murdok_core PUBLIC llama)

# Tool: murdok-hardware
add_executable(murdok-hardware
    tools/murdok-hardware/main.cpp
)
target_link_libraries(murdok-hardware PRIVATE murdok_core)

# Tool: murdok-bench
add_executable(murdok-bench
    tools/murdok-bench/main.cpp
)
target_link_libraries(murdok-bench PRIVATE murdok_core llama)

# Tool: murdok-server
add_executable(murdok-server
    tools/murdok-server/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp
)
set_target_properties(murdok-server PROPERTIES OUTPUT_NAME "murdok-serve")
target_link_libraries(murdok-server PRIVATE murdok_core)

# Tool: murdok CLI
add_executable(murdok
    tools/murdok/main.cpp
)
target_link_libraries(murdok PRIVATE murdok_core)

# Windows system libraries
if (WIN32)
    target_link_libraries(murdok_core PUBLIC psapi)
    target_link_libraries(murdok-bench PRIVATE psapi)
    target_link_libraries(murdok-server PRIVATE ws2_32 psapi)
    target_link_libraries(murdok PRIVATE ws2_32 psapi)
endif()
