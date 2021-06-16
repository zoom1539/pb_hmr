cmake_minimum_required(VERSION 3.0.0)



# CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# OpenCV
# find_package(OpenCV 3.4.8 REQUIRED PATHS ../extra/opencv348)
find_package(OpenCV REQUIRED)

# xtensor
find_package(xtensor REQUIRED)
message(${xtensor_INCLUDE_DIRS})
# json
find_package(nlohmann_json REQUIRED)

# #libtorch
# find_package(Torch REQUIRED PATHS /home/ubuntu/PLATFORM/3rdparty/libtorch130)
# message(STATUS "Torch library status:")
# message(STATUS "    version: ${Torch_VERSION}")
# message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
# message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

# if(NOT TORCH_FOUND)
#     message(FATAL_ERROR "Pytorch Not Found!")
# endif(NOT TORCH_FOUND)

set(LIBTORCH_ROOT /home/ubuntu/PLATFORM/3rdparty/libtorch130)
set(LIBTORCH_INCLUDE_DIRS ${LIBTORCH_ROOT}/include ${LIBTORCH_ROOT}/include/torch/csrc/api/include)
set(LIBTORCH_LIB_DIR ${LIBTORCH_ROOT}/lib)

# TensorRT
set(TENSORRT_ROOT /home/ubuntu/PLATFORM/3rdparty/TensorRT-7.0.0.11)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)


#
# aux_source_directory(. LIB_SRC)
file(GLOB_RECURSE LIB_SRC ./*.cpp ./*.cu)

# message(${LIB_SRC})
add_library(hmr SHARED ${LIB_SRC} )

target_include_directories(hmr 
            PUBLIC ${TENSORRT_INCLUDE_DIR}
            PUBLIC ${OpenCV_INCLUDE_DIRS}
            PUBLIC ${xtensor_INCLUDE_DIRS}
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extra/SMPLpp/include
            PUBLIC ${LIBTORCH_INCLUDE_DIRS})

target_link_libraries(hmr ${OpenCV_LIBS} 
                      ${TENSORRT_LIB_DIR}/libnvinfer.so 
                      ${CUDA_LIBRARIES}
                      xtensor
                      ${xtensor_blas_LIBRARIES}
                    #   ${TORCH_LIBRARIES}
                      ${LIBTORCH_LIB_DIR}/libtorch.so
                      nlohmann_json::nlohmann_json
                      stdc++fs)

# cd build 
# make install
install(FILES class_hmr.h DESTINATION include)
install(TARGETS hmr LIBRARY DESTINATION lib)
install(FILES ../hmr.engine DESTINATION lib)
