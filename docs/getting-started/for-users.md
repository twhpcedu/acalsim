# Getting Started for Users

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


---

- Author: Jen-Chien Chang \<jenchien@twhpcedu.org\>
- Date: 2024/08/05

([Back To Documentation Portal](/docs/README.md))

## Create New Projects From Templates

0. Initialize the ACALSim repository.
    ```shell
    cd /path/to/acalsim/
    make init
    ```
	> **Info**: It is recommended to do this whenever you update the ACALSim to a newer version.
1. Copy the template folder and rename it with your project name.
    ```shell
    cd src/
    cp -r "ProjectTemplate/" "YourProjectName/"
    ```
2. Update the CMake library name for your simulator project in the `libs/CMakeLists.txt` file within the folder you just created.
    ```cmake
    # Setup the libaray name
    ## You can modify the `template` below to any library name you prefer.
    set(APP_LIB_NAME template)
    ```
3. Follow the instructions in the user guide (link: TBD) to build your own simulator.

## Compile and Run Simulators

- Compile a CMake Target in Debug Mode:
	The target name for your simulator project is the same as the folder name under `src/` (e.g. `ProjectTemplate`).
	```shell
	cd /path/to/acalsim
	make debug TARGET=<TARGET_NAME>
	```
	The executable will be located in `build/debug/`. Run the simulator with the following command.
	```shell
	./build/debug/<TARGET_NAME>
	```
- Compile a CMake Target in Release Mode:
	```shell
	cd /path/to/acalsim
	make TARGET=<TARGET_NAME>
	```
	The executable will be located in `build/release/`. Run the simulator with the following command.
	```shell
	./build/release/<TARGET_NAME>
	```

## Third-Party Dependencies

- accellera-official/systemc: [3.0.0](https://github.com/accellera-official/systemc/tree/06ab23bc392cad78da8ab5d413fdc5d1a694dfe2)
- google/googletest: [1.14.0](https://github.com/google/googletest/tree/f8d7d77c06936315286eb55f8de22cd23c188571)
- andreiavrammsd/cpp-channel: [0.8.2](https://github.com/andreiavrammsd/cpp-channel/tree/ad66d30ae588d28c7f1a559d41316adff8c2d96f)
- CLIUtils/CLI11: [2.4.2](https://github.com/CLIUtils/CLI11/tree/6c7b07a878ad834957b98d0f9ce1dbe0cb204fc9)
- nlohmann/json: [3.11.3](https://github.com/nlohmann/json/tree/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)

---

([Back To Documentation Portal](/docs/README.md))
