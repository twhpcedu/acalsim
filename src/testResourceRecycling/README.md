<!--
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# testResourceRecycling

## Performance Profiling

### Using Callgrind

- Compile with CMake
	```shell
	$ cmake -B ./build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g -DPROFILE_CALLGRIND=ON
	$ cmake --build ./build/release -j $(nproc) --target testResourceRecycling
	```
	- To ensure optimal performance, please reset the Callgrind configuration with `-DPROFILE_CALLGRIND=OFF` once you have completed your profiling tasks. Leaving it enabled when not needed may impact performance.
- Execute for Profiling
	```shell
	$ valgrind --tool=callgrind --instr-atstart=no --collect-atstart=no ./build/release/testResourceRecycling
	```
- Translate Profiling Result
	```shell
	$ callgrind_annotate --tree=both --inclusive=no --auto=yes --show-percs=yes callgrind.out.${PID} > callgrind-profile.${PID}
	```
