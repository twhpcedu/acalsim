# About ACALSim

<!--
Copyright 2023-2026 Playlab/ACAL

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

## What is ACALSim

ACALSim is a high-speed, scalable C++ simulator framework designed to streamline the integration of simulators across diverse software architectures. As IC design projects become increasingly complex, the need for efficient component reuse and seamless co-simulation is paramount. ACALSim addresses these challenges by enabling users to effortlessly integrate simulators from various teams and open-source projects, ensuring smooth collaboration and efficient simulation workflows. This framework provides significant value by reducing the need for redundant component development and facilitating the collaborative efforts essential in modern large-scale IC design projects.

## About Playlab and AI Computing Architecture Lab (ACAL)

### From K-12 STEAM Education to Advanced AI Computing

Playlab, a collaborative learning group founded by Dr. Wei-Fen Lin in 2012, has evolved significantly over the past decade, adapting its focus to meet emerging educational and technological needs:

- **2012-2017: K-12 STEAM Education**
    - Provided courses and teacher training support for K-12 STEAM education
    - Formed a support group of college students to host educational events, workshops, and camps for K-12 students
- **2018-2022: Professional Development in AI and IC Design**
    - Shifted focus to offer professional short courses and training
    - Targeted professionals and college students in AI System Design and IC Design fields
- **Fall 2022: Launch of AI Computing Architecture Lab (ACAL)**
    - Created to develop open-source projects in high-performance computing
    - Aimed at promoting hands-on learning
    - Focused on bridging the skill gap between academia and industry requirements
- **Fall 2023: Introduction of ACALSim and Expanded Professional Courses**
    - Launched the ACALSim project to build a high-performance computing simulation environment
    - Designed for teaching and open research purposes
    - Leveraged ACALSim to offer more advanced professional courses in high-performance computing

## About Taiwan High-Performance Computing Education Association

The Taiwan High-Performance Computing Education Community is a grassroots initiative dedicated to making HPC education accessible to all and nurturing a new generation of HPC system designers. By fostering an open, collaborative, and inclusive environment, the community aims to:

- **Create an open learning ecosystem**: Develop and share free, high-quality HPC courses that combine theory with practical application.
- **Drive innovation through community**: Build and maintain an open-source research and learning platform fueled by diverse contributions from students, professionals, and enthusiasts.
- **Inspire growth**: Learn from successful open-source models like the [Rise: RISC-V Software Ecosystem](https://riseproject.dev) to create a vibrant HPC learning community.
- **Foster collaboration**: Bring together industry, academia, and students to share knowledge and drive collective progress.
- **Advance the HPC field**: Break down barriers between academia and industry to elevate the overall HPC landscape.

The community's commitment to open education, shared growth, and collaboration is driving innovation and accelerating the advancement of HPC. By joining this initiative, individuals can contribute to and benefit from a collective effort to push the boundaries of HPC knowledge and application.

## Key Features

ACALSim provides a comprehensive framework for building high-performance simulations:

### High Performance
- Multi-threading with specialized ThreadManager implementations
- Up to 14.56x speedup on large-scale simulations
- Object pooling for zero-allocation hot paths

### Flexible Architecture
- Generic event-driven simulation framework
- Support for heterogeneous system modeling
- SimPort and SimChannel for flexible communication
- Modular simulator composition

### Production Ready
- Comprehensive regression testing
- Extensive API documentation
- Well-tested example simulations
- Active development and maintenance

## License and Acknowledgment

This project is managed by the [Taiwan High-Performance Computing Education Association](https://twhpcedu.org/) and distributed under the Apache 2.0 License.

Copyright © 2023-2025 Playlab/ACAL

Licensed to Taiwan High-Performance Computing Education Association (TWHPCEDU) for educational use under the Apache 2.0 license.

## External Resources

- **GitHub Repository**: https://github.com/ACAL-Playlab/ACALSim
- **Issue Tracker**: https://github.com/ACAL-Playlab/ACALSim/issues
- **Documentation**: https://acalsim.playlab.tw
