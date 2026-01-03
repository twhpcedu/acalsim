# Copyright 2023-2026 Playlab/ACAL
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

ARG BASE_IMAGE_VER="2c7b3f97"
FROM playlabtw/acalsim-workspace:${BASE_IMAGE_VER}

ARG UID=1000
ARG GID=1000
ARG USERNAME="user"
ARG TZ="Asia/Taipei"

USER root

# install system packages
COPY ./dependency/apt-dependencies.txt /tmp/apt-dependencies.txt

# hadolint ignore=SC2046,DL3009
RUN apt-get -qq update && \
	apt-get -qq install $(cat "/tmp/apt-dependencies.txt") && \
	rm /tmp/apt-dependencies.txt

# install python libraries
COPY ./dependency/requirements.txt /tmp/requirements.txt
# hadolint ignore=DL3013
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
	rm /tmp/requirements.txt

# setup time zone
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# add non-root user account
RUN groupadd -o -g ${GID} "${USERNAME}" && \
	useradd -l -u ${UID} -m -s /bin/bash -g ${GID} "${USERNAME}" && \
	echo "${USERNAME} ALL = NOPASSWD: ALL" > "/etc/sudoers.d/${USERNAME}" && \
	chmod 0440 "/etc/sudoers.d/${USERNAME}" && \
	passwd -d "${USERNAME}"

# add scripts and setup permissions
COPY --chown=${UID}:${GID} ./scripts/.bashrc "/home/${USERNAME}/.bashrc"
COPY --chown=${UID}:${GID} ./scripts/start.sh /docker/start.sh
COPY --chown=${UID}:${GID} ./scripts/login.sh /docker/login.sh
COPY --chown=${UID}:${GID} ./scripts/startup.sh /usr/local/bin/startup

# hadolint ignore=DL4006
RUN dos2unix -ic "/home/${USERNAME}/.bashrc" | xargs dos2unix && \
	dos2unix -ic "/docker/start.sh" | xargs dos2unix && \
	dos2unix -ic "/docker/login.sh" | xargs dos2unix && \
	dos2unix -ic "/usr/local/bin/startup" | xargs dos2unix && \
	chmod +x "/usr/local/bin/startup"

# user account configuration
RUN mkdir -p "/home/${USERNAME}/.ssh" && \
	mkdir -p "/home/${USERNAME}/.vscode-server" && \
	mkdir -p "/home/${USERNAME}/projects"
RUN chown -R ${UID}:${GID} "/home/${USERNAME}" && \
	chown -R ${UID}:${GID} "/docker"

USER "${USERNAME}"

WORKDIR "/home/${USERNAME}"
