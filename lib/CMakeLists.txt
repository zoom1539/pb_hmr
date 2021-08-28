cmake_minimum_required(VERSION 3.0.0)

if (WIN32)
    MESSAGE(STATUS "Now is windows")
    # CUDA
    find_package(CUDA REQUIRED)
    MESSAGE(${CUDA_INCLUDE_DIRS})
    # OpenCV
    find_package(OpenCV 3.4.4 REQUIRED PATHS D:/3rd_party/opencv344/build_gpu/install)

    # TensorRT
    set(TENSORRT_ROOT D:/TensorRT-7.0.0.11)
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)

    #libtorch
    set(LIBTORCH_ROOT D:/3rd_party/libtorch120)
    set(LIBTORCH_INCLUDE_DIRS ${LIBTORCH_ROOT}/include ${LIBTORCH_ROOT}/include/torch/csrc/api/include)
    set(LIBTORCH_LIB_DIR ${LIBTORCH_ROOT}/lib)

    # xtensor
    find_package(xtl 0.7.2 REQUIRED PATHS D:/3rd_party/xtl-master/build/install)
    find_package(xtensor REQUIRED PATHS D:/3rd_party/xtensor-master/build/install)
    message(${xtensor_INCLUDE_DIRS})
    
    # json
    find_package(nlohmann_json REQUIRED PATHS D:/3rd_party/json3100/build/install)

    #
    # aux_source_directory(. LIB_SRC)
    file(GLOB_RECURSE LIB_SRC ./*.cpp ./*.cu)

    # message(${LIB_SRC})
    add_library(hmr SHARED ${LIB_SRC} )

    target_include_directories(hmr 
                PUBLIC ${CUDA_INCLUDE_DIRS}
                PUBLIC ${TENSORRT_INCLUDE_DIR}
                PUBLIC ${OpenCV_INCLUDE_DIRS}
                PUBLIC ${xtensor_INCLUDE_DIRS}
                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extra/SMPLpp/include
                PUBLIC ${LIBTORCH_INCLUDE_DIRS})

    target_link_libraries(hmr ${OpenCV_LIBS} 
                          ${TENSORRT_LIB_DIR}/nvinfer.lib 
                          ${CUDA_LIBRARIES}
                          xtensor
                          ${xtensor_blas_LIBRARIES}
                        #   ${TORCH_LIBRARIES}
                          ${LIBTORCH_LIB_DIR}/torch.lib
                          ${LIBTORCH_LIB_DIR}/c10.lib
                          nlohmann_json::nlohmann_json)
                        #  stdc++fs)

elseif (UNIX)
    # CUDA
    find_package(CUDA REQUIRED)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)


    # OpenCV
    find_package(OpenCV 3.4.8 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv348)
    # find_package(OpenCV REQUIRED)

    # xtensor
    find_package(xtl 0.7.2 REQUIRED PATHS /root/PLATFORM/packages/xtl/build/install)
    find_package(xtensor REQUIRED PATHS /root/PLATFORM/3rdparty/xtensor)
    message(${xtensor_INCLUDE_DIRS})
    # json
    find_package(nlohmann_json REQUIRED PATHS /root/PLATFORM/3rdparty/nlohmann_json)

    # #libtorch
    # find_package(Torch REQUIRED PATHS /home/ubuntu/PLATFORM/3rdparty/libtorch130)
    # message(STATUS "Torch library status:")
    # message(STATUS "    version: ${Torch_VERSION}")
    # message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
    # message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

    # if(NOT TORCH_FOUND)
    #     message(FATAL_ERROR "Pytorch Not Found!")
    # endif(NOT TORCH_FOUND)

    #libtorch
    set(LIBTORCH_ROOT /root/PLATFORM/3rdparty/libtorch131)
    set(LIBTORCH_INCLUDE_DIRS ${LIBTORCH_ROOT}/include ${LIBTORCH_ROOT}/include/torch/csrc/api/include)
    set(LIBTORCH_LIB_DIR ${LIBTORCH_ROOT}/lib)

    # TensorRT
    set(TENSORRT_ROOT /root/PLATFORM/3rdparty/TensorRT-7.0.0.11)
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
endif()

# cd build 
# make install
install(FILES class_hmr.h DESTINATION include)
install(TARGETS hmr LIBRARY DESTINATION lib)
install(FILES extra/SMPLpp/smpl_male.json DESTINATION lib)
install(FILES extra/SMPLpp/hmr.wts DESTINATION lib)
