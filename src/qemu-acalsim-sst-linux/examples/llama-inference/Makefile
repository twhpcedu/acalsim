# Makefile for LLAMA Inference Example

ROOTFS ?= /home/user/rootfs-full
INSTALL_DIR = $(ROOTFS)/apps/llama-inference

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
	cp README.md $(INSTALL_DIR)/
	chmod +x $(INSTALL_DIR)/llama_inference.py
	@echo "Installation complete"

deploy: install
	@echo "Rebuilding initramfs..."
	cd $(ROOTFS) && find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs-full.cpio.gz
	@echo "Deployment complete"
	@ls -lh /home/user/initramfs-full.cpio.gz

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
