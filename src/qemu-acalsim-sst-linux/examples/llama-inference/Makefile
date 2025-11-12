# Makefile for LLAMA Inference Example

ROOTFS ?= /home/user/rootfs
INSTALL_DIR = $(ROOTFS)/apps/llama-inference
INITRAMFS_BASE ?= /home/user/initramfs.cpio.gz
INITRAMFS_OUT ?= /home/user/initramfs-with-llama.cpio.gz

.PHONY: all install deploy clean help

all:
	@echo "Python application - no compilation needed"
	@echo "Use 'make install' to copy to rootfs"

install:
	@echo "Installing to rootfs..."
	mkdir -p $(INSTALL_DIR)
	cp llama_inference.py $(INSTALL_DIR)/
	cp llama_sst_backend.py $(INSTALL_DIR)/
	cp test_prompts.txt $(INSTALL_DIR)/
	cp run_sst.sh $(INSTALL_DIR)/
	cp run_qemu.sh $(INSTALL_DIR)/
	cp sst_config_llama.py $(INSTALL_DIR)/
	cp README.md $(INSTALL_DIR)/
	chmod +x $(INSTALL_DIR)/llama_inference.py
	chmod +x $(INSTALL_DIR)/run_sst.sh
	chmod +x $(INSTALL_DIR)/run_qemu.sh
	chmod +x $(INSTALL_DIR)/sst_config_llama.py
	@echo "Installation complete"
	@echo ""
	@echo "Files installed to: $(INSTALL_DIR)"
	@echo ""
	@echo "To use with QEMU:"
	@echo "  1. Build a complete rootfs with full Linux system first"
	@echo "  2. Run 'make deploy' to create initramfs"
	@echo "  3. Or use the existing initramfs.cpio.gz (run_qemu.sh does this)"

deploy: install
	@echo "Creating initramfs with LLAMA app..."
	@if [ ! -d "$(ROOTFS)/bin" ] || [ ! -d "$(ROOTFS)/sbin" ]; then \
		echo "ERROR: $(ROOTFS) does not contain a complete Linux filesystem"; \
		echo "       Missing /bin or /sbin directories"; \
		echo "       See PYTORCH_LLAMA_SETUP.md for building a complete rootfs"; \
		exit 1; \
	fi
	cd $(ROOTFS) && find . | cpio -o -H newc 2>/dev/null | gzip > $(INITRAMFS_OUT)
	@echo "Deployment complete"
	@ls -lh $(INITRAMFS_OUT)
	@echo ""
	@echo "To use this initramfs:"
	@echo "  export INITRAMFS=$(INITRAMFS_OUT)"
	@echo "  ./run_qemu.sh"

clean:
	@echo "Nothing to clean"

help:
	@echo "LLAMA Inference Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Check environment"
	@echo "  install   - Copy to rootfs"
	@echo "  deploy    - Install + rebuild initramfs"
	@echo "  clean     - Clean build artifacts"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make install      # Install to rootfs"
	@echo "  make deploy       # Build and deploy to initramfs"
