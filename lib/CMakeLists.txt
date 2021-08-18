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

    #
    # aux_source_directory(. LIB_SRC)
    file(GLOB_RECURSE LIB_SRC ./*.cpp ./*.cu)

    # message(${LIB_SRC})
    add_library(hmr SHARED ${LIB_SRC} )

    target_include_directories(hmr 
                PUBLIC ${CUDA_INCLUDE_DIRS}
                PUBLIC ${TENSORRT_INCLUDE_DIR}
                PUBLIC ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(hmr ${OpenCV_LIBS} 
                          ${TENSORRT_LIB_DIR}/nvinfer.lib 
                          ${CUDA_LIBRARIES})

elseif (UNIX)
    # CUDA
    find_package(CUDA REQUIRED)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)


    # OpenCV
    find_package(OpenCV 3.4.8 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv348)
    # find_package(OpenCV REQUIRED)

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
                PUBLIC ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(hmr ${OpenCV_LIBS} 
                          ${TENSORRT_LIB_DIR}/libnvinfer.so 
                          ${CUDA_LIBRARIES})
endif()

# cd build 
# make install
install(FILES class_hmr.h DESTINATION include)
install(TARGETS hmr LIBRARY DESTINATION lib)
install(FILES extra/hmr.wts DESTINATION lib)
