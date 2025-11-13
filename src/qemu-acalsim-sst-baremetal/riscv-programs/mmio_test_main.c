/*
 * Copyright 2023-2025 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * RISC-V MMIO Test Program - Using main()
 *
 * Tests the binary MMIO protocol with the SST device.
 * This version uses standard main() function with proper C runtime.
 */

#include <stdint.h>

// SST device MMIO registers (at 0x10200000)
#define SST_DEVICE_BASE 0x10200000

// Register offsets
#define SST_DATA_IN_OFFSET  0x00
#define SST_DATA_OUT_OFFSET 0x04
#define SST_STATUS_OFFSET   0x08
#define SST_CONTROL_OFFSET  0x0C

// MMIO register access
#define SST_DATA_IN  (*(volatile uint32_t*)(SST_DEVICE_BASE + SST_DATA_IN_OFFSET))
#define SST_DATA_OUT (*(volatile uint32_t*)(SST_DEVICE_BASE + SST_DATA_OUT_OFFSET))
#define SST_STATUS   (*(volatile uint32_t*)(SST_DEVICE_BASE + SST_STATUS_OFFSET))
#define SST_CONTROL  (*(volatile uint32_t*)(SST_DEVICE_BASE + SST_CONTROL_OFFSET))

// Status bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DATA_READY (1 << 1)
#define STATUS_ERROR      (1 << 2)

// Control bits
#define CONTROL_START (1 << 0)
#define CONTROL_RESET (1 << 1)

// Simple UART for debug output (QEMU virt machine)
#define UART_BASE 0x10000000
#define UART_TX   (*(volatile uint8_t*)UART_BASE)

// =============================================================================
// UART Helper Functions
// =============================================================================

void uart_putc(char c) { UART_TX = c; }

void uart_puts(const char* s) {
	while (*s) { uart_putc(*s++); }
}

void uart_puthex(uint32_t val) {
	const char hex[] = "0123456789ABCDEF";
	uart_puts("0x");
	for (int i = 7; i >= 0; i--) { uart_putc(hex[(val >> (i * 4)) & 0xF]); }
}

void uart_putdec(uint32_t val) {
	char buf[11];  // Max 10 digits + null
	int  i = 10;
	buf[i] = '\0';

	if (val == 0) {
		uart_putc('0');
		return;
	}

	while (val > 0 && i > 0) {
		buf[--i] = '0' + (val % 10);
		val /= 10;
	}

	uart_puts(&buf[i]);
}

// =============================================================================
// Trap Handler (required by crt0.S)
// =============================================================================

void trap_handler_c(uint32_t mcause, uint32_t mepc, uint32_t mtval) {
	uart_puts("\n[TRAP] Exception occurred!\n");
	uart_puts("  mcause: ");
	uart_puthex(mcause);
	uart_puts("\n  mepc:   ");
	uart_puthex(mepc);
	uart_puts("\n  mtval:  ");
	uart_puthex(mtval);
	uart_puts("\n");

	// Hang forever
	while (1) { asm volatile("wfi"); }
}

// =============================================================================
// Test Functions
// =============================================================================

int test_simple_write_read(void) {
	uart_puts("\n[TEST 1] Simple write/read\n");
	uart_puts("  Writing 0xDEADBEEF to SST_DATA_IN\n");

	// Write test value
	SST_DATA_IN = 0xDEADBEEF;

	uart_puts("  Triggering operation\n");
	SST_CONTROL = CONTROL_START;

	uart_puts("  Waiting for completion\n");
	// Wait for completion
	int timeout = 100000;
	while ((SST_STATUS & STATUS_BUSY) && timeout > 0) { timeout--; }

	if (timeout == 0) {
		uart_puts("  [TIMEOUT] Device did not respond\n");
		return 1;  // Failure
	}

	// Read result
	uint32_t result = SST_DATA_OUT;

	uart_puts("  Read result: ");
	uart_puthex(result);
	uart_puts("\n");

	if (result == 0xDEADBEEF) {
		uart_puts("  [PASS] Echo test passed\n");
		return 0;  // Success
	} else {
		uart_puts("  [FAIL] Expected 0xDEADBEEF\n");
		return 1;  // Failure
	}
}

int test_multiple_transactions(void) {
	uart_puts("\n[TEST 2] Multiple transactions\n");

	uint32_t test_values[] = {0x12345678, 0xCAFEBABE, 0xDEADC0DE, 0x0BADF00D, 0x1337BEEF};

	int passed = 0;
	int total  = sizeof(test_values) / sizeof(test_values[0]);

	for (int i = 0; i < total; i++) {
		uart_puts("  Transaction ");
		uart_putdec(i + 1);
		uart_puts(": ");
		uart_puthex(test_values[i]);
		uart_puts(" ... ");

		SST_DATA_IN = test_values[i];
		SST_CONTROL = CONTROL_START;

		int timeout = 100000;
		while ((SST_STATUS & STATUS_BUSY) && timeout > 0) { timeout--; }

		if (timeout == 0) {
			uart_puts("TIMEOUT\n");
			continue;
		}

		uint32_t result = SST_DATA_OUT;

		if (result == test_values[i]) {
			uart_puts("PASS\n");
			passed++;
		} else {
			uart_puts("FAIL (got ");
			uart_puthex(result);
			uart_puts(")\n");
		}
	}

	uart_puts("  Result: ");
	uart_putdec(passed);
	uart_puts("/");
	uart_putdec(total);
	uart_puts(" passed\n");

	if (passed == total) {
		uart_puts("  [PASS] All transactions passed\n");
		return 0;  // Success
	} else {
		uart_puts("  [FAIL] Some transactions failed\n");
		return 1;  // Failure
	}
}

int test_status_register(void) {
	uart_puts("\n[TEST 3] Status register\n");
	uart_puts("  Reading initial status\n");

	uint32_t status = SST_STATUS;
	uart_puts("  Initial status: ");
	uart_puthex(status);
	uart_puts("\n");

	uart_puts("  Status bits:\n");
	uart_puts("    BUSY:       ");
	uart_puts((status & STATUS_BUSY) ? "1" : "0");
	uart_puts("\n");
	uart_puts("    DATA_READY: ");
	uart_puts((status & STATUS_DATA_READY) ? "1" : "0");
	uart_puts("\n");
	uart_puts("    ERROR:      ");
	uart_puts((status & STATUS_ERROR) ? "1" : "0");
	uart_puts("\n");

	uart_puts("  [PASS] Status register readable\n");
	return 0;  // Success
}

int test_control_register(void) {
	uart_puts("\n[TEST 4] Control register\n");
	uart_puts("  Testing RESET bit\n");

	SST_CONTROL = CONTROL_RESET;
	uart_puts("  Reset issued\n");

	int timeout = 10000;
	while ((SST_STATUS & STATUS_BUSY) && timeout > 0) { timeout--; }

	uart_puts("  [PASS] Control register writable\n");
	return 0;  // Success
}

// =============================================================================
// Main Function
// =============================================================================

int main(int argc, char* argv[]) {
	// Suppress unused parameter warnings
	(void)argc;
	(void)argv;

	uart_puts("===========================================\n");
	uart_puts("  QEMU-SST: Binary MMIO Protocol Test\n");
	uart_puts("  Using standard main() function\n");
	uart_puts("===========================================\n");
	uart_puts("\n");
	uart_puts("Device base address: ");
	uart_puthex(SST_DEVICE_BASE);
	uart_puts("\n");

	// Run tests and track results
	int failures = 0;

	failures += test_simple_write_read();
	failures += test_multiple_transactions();
	failures += test_status_register();
	failures += test_control_register();

	// Print summary
	uart_puts("\n===========================================\n");
	uart_puts("  Test Summary\n");
	uart_puts("===========================================\n");
	uart_puts("  Total failures: ");
	uart_putdec(failures);
	uart_puts("\n");

	if (failures == 0) {
		uart_puts("  [SUCCESS] All tests passed!\n");
	} else {
		uart_puts("  [FAILURE] Some tests failed!\n");
	}
	uart_puts("===========================================\n");

	// Return number of failures as exit code
	return failures;
}
