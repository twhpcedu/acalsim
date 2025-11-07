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

# Get the directory name
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set the executable name
set(EXE_NAME ${DIR_NAME})

# Declare the executable
add_executable(${EXE_NAME} main.cc) ## Modify the executable name `AcalSim.cc` to your project name

# Include directories for the target
target_include_directories(${EXE_NAME} PRIVATE include)

# Add subdirectories
add_subdirectory(libs)

# Link libraries to the executable
target_link_libraries(${EXE_NAME} ACALSim::testSimPort) ## Modify `template_lib` here if you modify the libs name inside `libs/CMakeLists.txt`
