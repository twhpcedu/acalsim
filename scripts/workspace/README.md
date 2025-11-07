# ACALSim Docker Workspace

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

A dockerized [Ubuntu 22.04](https://hub.docker.com/_/ubuntu/) workspace for ACALSim-based project development.

## Table of Contents <!-- omit in toc -->

- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Workspace Management Script](#workspace-management-script)
    - [Common Operations](#common-operations)
    - [Working Directory](#working-directory)
- [Docker Volumes](#docker-volumes)
- [Main Package Versions](#main-package-versions)
- [Personalization](#personalization)
    - [Configuring The Username](#configuring-the-username)
    - [Adding System Packages to the Workspace](#adding-system-packages-to-the-workspace)
    - [Adding Python Packages to the Workspace](#adding-python-packages-to-the-workspace)

## Getting Started

### Prerequisites

- The Docker Engine is required for launching the workspace. Please refer to [the official documentation](https://docs.docker.com/desktop/) for more details.
- Ensure that your environment has an active internet connection, as the workspace will be downloaded from DockerHub.

### Workspace Management Script

The [`run`](./run) script facilitates easy management of the workspace. If you are using a Windows system, please execute the following commands in a Bash-like shell such as [Git Bash](https://git-scm.com/download/win).

```shell
bash /path/to/acalsim/scripts/workspace/run
```

You will see all available subcommands provided by the script.

```text
    This script will help you manage the Docker Workspace for ACALSim project development.
    You can execute this script with the following options.

    pull            : pull the latest official image from DockerHub
    build           : build a new image on this machine
    start           : pull and enter the workspace
    stop            : stop and exit the workspace
    prune           : remove the docker image
    repull          : remove the existing image and pull the latest one to apply new changes
    rebuild         : remove the existing image and build a new one to apply new changes
```

### Common Operations

1. Pull the workspace from the [DockerHub](https://hub.docker.com/repository/docker/playlabtw/acalsim-workspace/general).

    ```shell
    ./run pull
    ```

2. Launch and enter the workspace.

    ```shell
    ./run start
    ```

3. Terminate the workspace when your work finished.

    ```shell
    ./run stop
    ```

### Working Directory

The workspace will mount the top-level repository of ACALSim from the host machine into the home directory of the workspace. The mounting behavior is as follows:

- **Standalone ACALSim Repository**: If ACALSim is a standalone repository, it will be mounted at `~/acalsim` within the workspace.
- **ACALSim as a Submodule**: If ACALSim is included as a submodule within `YourProject`, the root directory of `YourProject` will be mounted at `~/YourProject` in the workspace.

## Docker Volumes

The workspace uses several Docker volumes to support the following purposes:

1. File sharing between the workspace and the host operating system.
2. Retention of files after the workspace is terminated.

All docker volumes including the following directories in the workspace:

1. `~/acalsim/`: The repository users develop with the workspace. Note that the name will be changed according to the top-level repository found by the `run` script.
2. `~/.config/`: Contains user-specific configuration files for customizing tools in the workspace.
3. `~/.ssh/`: Stores SSH keys for secure access to remote servers and repositories. Users can either generate a new key pair in the workspace or copy a currently used one from your host machine into this folder.
4. `~/.cache/pre-commit/`: Holds cache files for the pre-commit framework, improving performance by reducing repeated downloads.
5. `~/.vscode-server/`: Contains files used by the VSCode remote server to streamline remote development setups.

The mounting directories of these volumes on the host operating system will be displayed whenever users enter the workspace.

## Main Package Versions

- Libtorch: 2.2.0
- CMake: 3.25.0
- PyTorch: 2.2.0
- ONNX: 1.14.0
- ONNXRuntime: 1.17.1
- NumPy: 1.26.4

## Personalization

Users can customize the workspace to meet specific requirements by modifying certain files and rebuilding the workspace.

### Configuring The Username

1. Modify the `CONTAINER_USERNAME` in the [`run`](./run) script.
2. Rebuild the workspace using:

    ```shell
    ./run rebuild
    ```

3. Start and enter the workspace with:

    ```shell
    ./run start
    ```

### Adding System Packages to the Workspace

1. List the desired packages into [`dependency/apt-dependencies.txt`](./dependency/apt-dependencies.txt) file, either one package per line or multiple packages on a single line.
2. Rebuild the workspace using:

    ```shell
    ./run rebuild
    ```

3. Start and enter the workspace with:

    ```shell
    ./run start
    ```

### Adding Python Packages to the Workspace

1. Specify the packages and versions into [`dependency/requirements.txt`](./dependency/requirements.txt) file.
2. Rebuild the workspace using:

    ```shell
    ./run rebuild
    ```

3. Start and enter the workspace with:

    ```shell
    ./run start
    ```
