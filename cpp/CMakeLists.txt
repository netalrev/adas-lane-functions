cmake_minimum_required(VERSION 3.18)
project(adas_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Set ONNXRUNTIME_ROOT to the unpacked ORT SDK directory, e.g.:
#   cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.16.3
#
# Download from: https://github.com/microsoft/onnxruntime/releases
# Required layout:
#   ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h
#   ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so (Linux) or onnxruntime.dll (Windows)

if(NOT DEFINED ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is not set.\n"
    "Download the ORT SDK from https://github.com/microsoft/onnxruntime/releases "
    "and pass: cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-X.Y.Z"
  )
endif()

find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  REQUIRED
)

find_library(ORT_LIB
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  REQUIRED
)

message(STATUS "ORT include : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT library : ${ORT_LIB}")

# ── nlohmann/json (header-only, fetched automatically) ────────────────────────
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Library: adas_inference ────────────────────────────────────────────────────
add_library(adas_inference STATIC
  src/adas_inference.cpp
)

target_include_directories(adas_inference
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(adas_inference
  PUBLIC
    ${ORT_LIB}
    nlohmann_json::nlohmann_json
)

target_compile_options(adas_inference PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3>
)

# ── Executable: adas_infer ─────────────────────────────────────────────────────
add_executable(adas_infer src/main.cpp)

target_link_libraries(adas_infer PRIVATE adas_inference)

target_compile_options(adas_infer PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3>
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS adas_infer DESTINATION bin)
install(TARGETS adas_inference DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# ── Quick-start build instructions (cmake --build help) ──────────────────────
# 1. Download ORT SDK for your platform from:
#       https://github.com/microsoft/onnxruntime/releases
#
# 2. Configure:
#       mkdir build && cd build
#       cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.16.3
#
# 3. Build:
#       cmake --build . --config Release -j$(nproc)
#
# 4. Run (inference on a single JSON frame):
#       ./adas_infer --model ../../outputs/models/mf_transformer_int8.onnx \
#                    --input frame.json
