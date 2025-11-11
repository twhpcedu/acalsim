#
# Makefile for VirtIO SST Kernel Driver
#
# Copyright 2023-2025 Playlab/ACAL
#

# Module name
obj-m := virtio-sst.o

# Source files
virtio-sst-objs := sst-virtio.o

# Include paths (for sst-protocol.h)
ccflags-y := -I$(src)/../virtio-device

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Default target - build module
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean build artifacts
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f Module.symvers modules.order

# Install module
install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -A

# Uninstall module
uninstall:
	rm -f $(KDIR)/extra/virtio-sst.ko
	depmod -A

# Load module
load:
	sudo insmod virtio-sst.ko

# Unload module
unload:
	sudo rmmod virtio-sst

# Reload module (unload + load)
reload: unload load

# Show module info
info:
	modinfo virtio-sst.ko

# Show kernel log
log:
	dmesg | tail -50 | grep virtio-sst

# Help
help:
	@echo "VirtIO SST Kernel Driver Build Targets:"
	@echo "  all        - Build kernel module (default)"
	@echo "  clean      - Clean build artifacts"
	@echo "  install    - Install module to system"
	@echo "  uninstall  - Remove module from system"
	@echo "  load       - Load module into kernel"
	@echo "  unload     - Unload module from kernel"
	@echo "  reload     - Reload module (unload + load)"
	@echo "  info       - Show module information"
	@echo "  log        - Show kernel log for module"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR       - Kernel build directory (default: /lib/modules/\$$(uname -r)/build)"
	@echo ""
	@echo "Example:"
	@echo "  make                    # Build module"
	@echo "  make KDIR=/path/to/kernel  # Build for specific kernel"
	@echo "  sudo make install       # Install to system"
	@echo "  sudo make load          # Load into kernel"

.PHONY: all clean install uninstall load unload reload info log help
