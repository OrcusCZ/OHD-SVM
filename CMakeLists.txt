CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)
PROJECT(OHD-SVM)

FIND_PACKAGE(CUDA REQUIRED)

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")

IF(CMAKE_COMPILER_IS_GNUCC)
    STRING(REGEX REPLACE ".*/" "" C ${CUDA_HOST_COMPILER})
    IF(${C} STREQUAL "ccache")
        MESSAGE(WARNING "CUDA_HOST_COMPILER is set to: ${CUDA_HOST_COMPILER}, changing to /usr/bin/g++")
        SET(CUDA_HOST_COMPILER "/usr/bin/g++")
    ENDIF()
ENDIF()

ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)

SET(sources
    dev_vars.h
    csr.h
    csr.cu
    calc_x2.h
    calc_x2.cu
    check_cache.h
    kernels_select_ws.h
    kernels_klocal.h
    kernels_smo.h
    ohdSVM.cu
    ohdSVM.h
    debug.h
    debug.cpp
    cudaerror.h
    cuda_utils.h
) 

CUDA_ADD_LIBRARY(ohd-svm STATIC ${sources})
CUDA_ADD_CUBLAS_TO_TARGET(ohd-svm)
TARGET_LINK_LIBRARIES(ohd-svm ${CUDA_CUDA_LIBRARY})
SET_PROPERTY(TARGET ohd-svm PROPERTY CXX_STANDARD 11)