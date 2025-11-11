#
# Makefile for SST Test Applications
#
# Copyright 2023-2025 Playlab/ACAL
#

# Compiler and flags
CC ?= gcc
CFLAGS := -Wall -Wextra -O2 -I../../virtio-device
LDFLAGS :=

# Cross-compilation support
ifdef CROSS_COMPILE
    CC := $(CROSS_COMPILE)gcc
endif

# Targets
TARGETS := sst-test

# Default target
all: $(TARGETS)

# Build sst-test
sst-test: sst-test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGETS) *.o

# Install to rootfs
install: $(TARGETS)
	mkdir -p $(DESTDIR)/apps
	cp -f $(TARGETS) $(DESTDIR)/apps/
	chmod +x $(DESTDIR)/apps/*

# Help
help:
	@echo "SST Test Applications Build"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all test applications (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install to rootfs (set DESTDIR)"
	@echo ""
	@echo "Variables:"
	@echo "  CC             - C compiler (default: gcc)"
	@echo "  CFLAGS         - Compiler flags"
	@echo "  CROSS_COMPILE  - Cross-compiler prefix (e.g., riscv64-linux-gnu-)"
	@echo "  DESTDIR        - Installation root directory"
	@echo ""
	@echo "Examples:"
	@echo "  make                                      # Build for host"
	@echo "  make CROSS_COMPILE=riscv64-linux-gnu-   # Cross-compile for RISC-V"
	@echo "  make install DESTDIR=/path/to/rootfs     # Install to rootfs"

.PHONY: all clean install help
