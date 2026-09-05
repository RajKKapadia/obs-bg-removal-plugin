cmake_minimum_required(VERSION 3.24)
project(obs-rmbg VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/.deps/onnxruntime-linux-x64-1.29.0" CACHE PATH "Extracted ONNX Runtime SDK")
set(RMBG_CUDA_LIBRARY_DIR "" CACHE PATH "Optional private CUDA/cuDNN library directory for Linux")
option(RMBG_BUILD_PLUGIN "Build the OBS filter" ON)
option(RMBG_INSTALL_MODEL "Include your downloaded model in a personal install" OFF)

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(ONNXRuntime SHARED IMPORTED)
set_target_properties(ONNXRuntime PROPERTIES IMPORTED_LOCATION "${ORT_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
find_package(Threads REQUIRED)

add_library(rmbg-core STATIC src/model.cpp src/worker.cpp)
target_include_directories(rmbg-core PUBLIC src)
target_link_libraries(rmbg-core PUBLIC ONNXRuntime Threads::Threads)
if(RMBG_CUDA_LIBRARY_DIR)
  # Direct dependencies ensure ORT's later dlopen uses these matching versions.
  # Keep these handles loaded for the full lifetime of the plugin and its worker.
  target_link_libraries(rmbg-core PUBLIC "-Wl,--no-as-needed"
    "${RMBG_CUDA_LIBRARY_DIR}/libcudart.so.12"
    "${RMBG_CUDA_LIBRARY_DIR}/libcublasLt.so.12"
    "${RMBG_CUDA_LIBRARY_DIR}/libcublas.so.12"
    "${RMBG_CUDA_LIBRARY_DIR}/libcudnn.so.9" "-Wl,--as-needed")
endif()
target_compile_options(rmbg-core PRIVATE -Wall -Wextra -Wpedantic)

if(RMBG_BUILD_PLUGIN)
  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/.deps/sysroot/usr")
  find_package(libobs REQUIRED CONFIG)
  if(libobs_VERSION VERSION_LESS 30)
    message(FATAL_ERROR "OBS 30 or newer development files are required")
  endif()
  add_library(obs-rmbg MODULE src/plugin.cpp)
  target_link_libraries(obs-rmbg PRIVATE rmbg-core OBS::libobs)
  target_compile_options(obs-rmbg PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(obs-rmbg PROPERTIES PREFIX "" BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${RMBG_CUDA_LIBRARY_DIR}" INSTALL_RPATH "$ORIGIN;$ORIGIN/cuda")
  install(TARGETS obs-rmbg LIBRARY DESTINATION obs-rmbg/bin/64bit)
  install(DIRECTORY data/ DESTINATION obs-rmbg/data PATTERN models EXCLUDE)
  install(DIRECTORY "${ONNXRUNTIME_ROOT}/lib/" DESTINATION obs-rmbg/bin/64bit
          FILES_MATCHING PATTERN "libonnxruntime*.so*")
  install(FILES "${ONNXRUNTIME_ROOT}/LICENSE" "${ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt"
          DESTINATION obs-rmbg/data/licenses)
  install(FILES LICENSE COPYING DESTINATION obs-rmbg/data/licenses/plugin)
  if(RMBG_CUDA_LIBRARY_DIR)
    install(DIRECTORY "${RMBG_CUDA_LIBRARY_DIR}/" DESTINATION obs-rmbg/bin/64bit/cuda FILES_MATCHING PATTERN "*.so*")
    install(DIRECTORY "${RMBG_CUDA_LIBRARY_DIR}/../licenses/" DESTINATION obs-rmbg/data/licenses/cuda OPTIONAL)
  endif()
  if(RMBG_INSTALL_MODEL)
    install(FILES data/models/rmbg-1.4.onnx DESTINATION obs-rmbg/data/models)
  endif()
endif()

find_package(PNG REQUIRED)
add_executable(rmbg-image tools/rmbg-image.cpp)
target_link_libraries(rmbg-image PRIVATE rmbg-core PNG::PNG)

include(CTest)
if(BUILD_TESTING)
  add_executable(rmbg-tests tests/core-tests.cpp)
  target_link_libraries(rmbg-tests PRIVATE rmbg-core)
  add_test(NAME rmbg-core COMMAND rmbg-tests)
  if(RMBG_BUILD_PLUGIN)
    add_executable(obs-rmbg-smoke tests/obs-smoke.cpp)
    target_link_libraries(obs-rmbg-smoke PRIVATE OBS::libobs PNG::PNG Threads::Threads)
  endif()
endif()
