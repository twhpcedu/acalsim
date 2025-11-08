TARGET := ProjectTemplate

ACALSIM_VERBOSE := OFF
ACALSIM_STATISTICS := OFF
MT_DEBUG := OFF
NO_LOGS := OFF
BUILD_SHARED_LIBS := ON

CMAKE_FLAGS += -DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS)
CMAKE_FLAGS += -DACALSIM_VERBOSE=$(ACALSIM_VERBOSE) -DMT_DEBUG=$(MT_DEBUG) -DNO_LOGS=$(NO_LOGS) -DACALSIM_STATISTICS=$(ACALSIM_STATISTICS)

BUILD_DIR := build
BUILD_DEBUG_DIR := $(BUILD_DIR)/debug
BUILD_REL_WITH_DEB_INFO_DIR := $(BUILD_DIR)/rel_with_deb_info
BUILD_RELEASE_DIR := $(BUILD_DIR)/release

MAKEFLAGS += --no-print-directory
CMAKE_BUILD_PARALLEL_LEVEL ?= $(shell nproc)

# Compile libraries and executables
.PHONY: all comp_cmd debug rel_with_deb_info release

all: release

comp_cmd: # Generate compile_commands.json in the root directory of the project
	@cmake -B $(BUILD_DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) -DCMAKE_EXPORT_COMPILE_COMMANDS=1
	@test -f "compile_commands.json" || ln -s "$(BUILD_DEBUG_DIR)/compile_commands.json" "compile_commands.json"

debug: # Build targets for debugging
	@cmake -B $(BUILD_DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS) -DCMAKE_EXPORT_COMPILE_COMMANDS=1
	@cmake --build $(BUILD_DEBUG_DIR) -j $(CMAKE_BUILD_PARALLEL_LEVEL) --target $(TARGET)
	@test -f "compile_commands.json" || ln -s "$(BUILD_DEBUG_DIR)/compile_commands.json" "compile_commands.json"

rel_with_deb_info: # Build targets for debugging with compiler optimization
	@cmake -B $(BUILD_REL_WITH_DEB_INFO_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_FLAGS)
	@cmake --build $(BUILD_REL_WITH_DEB_INFO_DIR) -j $(CMAKE_BUILD_PARALLEL_LEVEL) --target $(TARGET)

release: # Build targets for release
	@cmake -B $(BUILD_RELEASE_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	@cmake --build $(BUILD_RELEASE_DIR) -j $(CMAKE_BUILD_PARALLEL_LEVEL) --target $(TARGET)

# Auxiliary tools
.PHONY: test regression pre-commit

test: regression

regression: # Run regression test
	@python3 scripts/regression.py

pre-commit: # Run pre-commit against the whole repository
	@pre-commit run --all

# Repository management
.PHONY: init clean

init: clean # Initialize the repository
	@rm -rf third-party/*/
	@git submodule update --init --recursive --depth 1
	@pre-commit install

clean: # Clean built files
	@rm -rf $(BUILD_DIR)
