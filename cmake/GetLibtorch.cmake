# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# // clang-format off

include(FetchContent)

macro(get_libtorch)
    if(NOT Torch_FOUND)
        find_package(Torch QUIET)
    endif(NOT Torch_FOUND)

    if(NOT Torch_FOUND)
        if(
            CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
            OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"
        )
            set(LIBTORCH_URL
                "https://file.playlab.tw/libtorch-aarch64-2.2.0+cpu.zip"
            )
        else()
            set(LIBTORCH_URL
                "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip"
            )
        endif()

        fetchcontent_declare(
            libtorch
            DOWNLOAD_EXTRACT_TIMESTAMP
            URL ${LIBTORCH_URL}
        )

        fetchcontent_makeavailable(libtorch)
        list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
        find_package(Torch REQUIRED)
    endif(NOT Torch_FOUND)
endmacro(get_libtorch)

macro(build_libtorch)
    # include(FetchContent)

    # set(BUILD_CAFFE2 ON CACHE BOOL "" FORCE)
    # set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    # set(BUILD_TEST OFF CACHE BOOL "" FORCE)
    # set(USE_CUDA OFF CACHE BOOL "" FORCE)
    # set(USE_CUDNN OFF CACHE BOOL "" FORCE)
    # set(USE_ROCM OFF CACHE BOOL "" FORCE)
    # set(USE_OPENCL OFF CACHE BOOL "" FORCE)

    # set(CMAKE_VERBOSE_MAKEFILE ON)

    # FetchContent_Declare(
    #     pytorch
    #     GIT_REPOSITORY https://github.com/pytorch/pytorch.git
    #     GIT_TAG v2.2.0
    #     GIT_PROGRESS TRUE
    #     GIT_SHALLOW TRUE
    # )

    # FetchContent_MakeAvailable(pytorch)
endmacro(build_libtorch)
