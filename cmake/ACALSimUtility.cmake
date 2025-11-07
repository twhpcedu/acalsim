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

macro(subdirlist result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child})
            set(dirlist ${dirlist} ${child})
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro(subdirlist)

macro(realpath result input)
    get_filename_component(${result} ${input} REALPATH)
endmacro(realpath)

function(add_target_compile_definitions target_name)
    foreach(definition IN LISTS ARGN)
        if(${definition})
            target_compile_definitions(${target_name} PUBLIC ${definition})
        endif()
    endforeach()
endfunction(add_target_compile_definitions)

macro(option_summary)
    message(
        STATUS
        "=============================================================="
    )
    message(STATUS "Settings to build ACALSim ${CMAKE_PROJECT_VERSION}")
    message(
        STATUS
        "--------------------------------------------------------------"
    )

    message(STATUS "ACALSIM_VERBOSE = ${ACALSIM_VERBOSE}")
    message(STATUS "MT_DEBUG = ${MT_DEBUG}")
    message(STATUS "NO_LOGS = ${NO_LOGS}")
    message(STATUS "ACALSIM_STATISTICS = ${ACALSIM_STATISTICS}")
    message(
        STATUS
        "--------------------------------------------------------------"
    )

    message(
        STATUS
        "CMAKE_CXX_STANDARD_REQUIRED = ${CMAKE_CXX_STANDARD_REQUIRED}"
    )
    message(STATUS "CMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}")
    message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
    message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
    message(STATUS "BUILD_SHARED_LIBS = ${BUILD_SHARED_LIBS}")
    message(
        STATUS
        "--------------------------------------------------------------"
    )

    message(STATUS "CMAKE_VERBOSE_MAKEFILE = ${CMAKE_VERBOSE_MAKEFILE}")
    message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
    message(
        STATUS
        "CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
    message(
        STATUS
        "--------------------------------------------------------------"
    )

    message(STATUS "Torch_DIR=${Torch_DIR}")

    message(
        STATUS
        "=============================================================="
    )
endmacro(option_summary)
