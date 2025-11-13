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

/**
 * Multi-Device Test Program
 *
 * This program tests the QEMU-SST integration with two devices:
 * - Device 1: Echo device at 0x10200000
 * - Device 2: Compute device at 0x10300000
 *
 * It demonstrates:
 * 1. Communication with multiple SST devices
 * 2. Device-specific operations (echo vs. compute)
 * 3. Inter-device communication via SST
 */

#include <stdint.h>

// UART for console output
#define UART_BASE 0x10000000
#define UART_THR  (*(volatile uint8_t*)(UART_BASE + 0x00))

// Echo Device (Device 1) at 0x10200000
#define ECHO_DEVICE_BASE 0x10200000
#define ECHO_DATA_IN     (*(volatile uint32_t*)(ECHO_DEVICE_BASE + 0x00))
#define ECHO_DATA_OUT    (*(volatile uint32_t*)(ECHO_DEVICE_BASE + 0x04))
#define ECHO_STATUS      (*(volatile uint32_t*)(ECHO_DEVICE_BASE + 0x08))
#define ECHO_CONTROL     (*(volatile uint32_t*)(ECHO_DEVICE_BASE + 0x0C))

// Compute Device (Device 2) at 0x10300000
#define COMPUTE_DEVICE_BASE 0x10300000
#define COMPUTE_OPERAND_A   (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x00))
#define COMPUTE_OPERAND_B   (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x04))
#define COMPUTE_OPERATION   (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x08))
#define COMPUTE_RESULT      (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x0C))
#define COMPUTE_STATUS      (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x10))
#define COMPUTE_CONTROL     (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x14))
#define COMPUTE_PEER_OUT    (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x18))
#define COMPUTE_PEER_IN     (*(volatile uint32_t*)(COMPUTE_DEVICE_BASE + 0x1C))

// Compute operations
#define OP_ADD 0
#define OP_SUB 1
#define OP_MUL 2
#define OP_DIV 3

// Status bits
#define STATUS_BUSY       0x01
#define STATUS_READY      0x02
#define STATUS_ERROR      0x04
#define STATUS_PEER_READY 0x08

// Control bits
#define CONTROL_RESET   0x01
#define CONTROL_TRIGGER 0x02

// Simple UART print functions
void uart_putc(char c) { UART_THR = c; }

void uart_puts(const char* s) {
	while (*s) { uart_putc(*s++); }
}

void uart_puthex(uint32_t val) {
	const char hex[] = "0123456789ABCDEF";
	uart_puts("0x");
	for (int i = 28; i >= 0; i -= 4) { uart_putc(hex[(val >> i) & 0xF]); }
}

void wait_cycles(uint32_t cycles) {
	for (volatile uint32_t i = 0; i < cycles; i++) { asm volatile("nop"); }
}

// Test echo device
int test_echo_device(void) {
	uart_puts("[TEST 1] Echo Device Test\n");

	// Write data and read it back
	uint32_t test_val = 0xCAFEBABE;
	ECHO_DATA_IN      = test_val;
	ECHO_CONTROL      = CONTROL_TRIGGER;

	// Wait for ready
	uint32_t timeout = 10000;
	while ((ECHO_STATUS & STATUS_READY) == 0 && timeout-- > 0) { wait_cycles(10); }

	if (timeout == 0) {
		uart_puts("  [FAIL] Echo device timeout\n");
		return 1;
	}

	uint32_t result = ECHO_DATA_OUT;
	if (result == test_val) {
		uart_puts("  [PASS] Echo device: ");
		uart_puthex(result);
		uart_puts("\n");
		return 0;
	} else {
		uart_puts("  [FAIL] Echo mismatch: expected ");
		uart_puthex(test_val);
		uart_puts(" got ");
		uart_puthex(result);
		uart_puts("\n");
		return 1;
	}
}

// Test compute device arithmetic
int test_compute_device(void) {
	uart_puts("[TEST 2] Compute Device Test\n");
	int failures = 0;

	// Test 1: Addition (42 + 58 = 100)
	COMPUTE_OPERAND_A = 42;
	COMPUTE_OPERAND_B = 58;
	COMPUTE_OPERATION = OP_ADD;
	COMPUTE_CONTROL   = CONTROL_TRIGGER;

	// Wait for computation
	uint32_t timeout = 10000;
	while ((COMPUTE_STATUS & STATUS_READY) == 0 && timeout-- > 0) {
		wait_cycles(100);  // Computation takes longer
	}

	if (timeout == 0) {
		uart_puts("  [FAIL] Compute timeout (ADD)\n");
		failures++;
	} else {
		uint32_t result = COMPUTE_RESULT;
		if (result == 100) {
			uart_puts("  [PASS] ADD: 42 + 58 = ");
			uart_puthex(result);
			uart_puts("\n");
		} else {
			uart_puts("  [FAIL] ADD: expected 100, got ");
			uart_puthex(result);
			uart_puts("\n");
			failures++;
		}
	}

	// Test 2: Multiplication (12 * 5 = 60)
	COMPUTE_OPERAND_A = 12;
	COMPUTE_OPERAND_B = 5;
	COMPUTE_OPERATION = OP_MUL;
	COMPUTE_CONTROL   = CONTROL_TRIGGER;

	timeout = 10000;
	while ((COMPUTE_STATUS & STATUS_READY) == 0 && timeout-- > 0) { wait_cycles(100); }

	if (timeout == 0) {
		uart_puts("  [FAIL] Compute timeout (MUL)\n");
		failures++;
	} else {
		uint32_t result = COMPUTE_RESULT;
		if (result == 60) {
			uart_puts("  [PASS] MUL: 12 * 5 = ");
			uart_puthex(result);
			uart_puts("\n");
		} else {
			uart_puts("  [FAIL] MUL: expected 60, got ");
			uart_puthex(result);
			uart_puts("\n");
			failures++;
		}
	}

	// Test 3: Division (100 / 4 = 25)
	COMPUTE_OPERAND_A = 100;
	COMPUTE_OPERAND_B = 4;
	COMPUTE_OPERATION = OP_DIV;
	COMPUTE_CONTROL   = CONTROL_TRIGGER;

	timeout = 10000;
	while ((COMPUTE_STATUS & STATUS_READY) == 0 && timeout-- > 0) { wait_cycles(100); }

	if (timeout == 0) {
		uart_puts("  [FAIL] Compute timeout (DIV)\n");
		failures++;
	} else {
		uint32_t result = COMPUTE_RESULT;
		if (result == 25) {
			uart_puts("  [PASS] DIV: 100 / 4 = ");
			uart_puthex(result);
			uart_puts("\n");
		} else {
			uart_puts("  [FAIL] DIV: expected 25, got ");
			uart_puthex(result);
			uart_puts("\n");
			failures++;
		}
	}

	return failures;
}

// Test inter-device communication
int test_inter_device_communication(void) {
	uart_puts("[TEST 3] Inter-Device Communication Test\n");

	// Step 1: Write a value to echo device
	uint32_t echo_val = 0xDEAD1337;
	ECHO_DATA_IN      = echo_val;
	ECHO_CONTROL      = CONTROL_TRIGGER;

	// Wait for echo device to be ready
	uint32_t timeout = 10000;
	while ((ECHO_STATUS & STATUS_READY) == 0 && timeout-- > 0) { wait_cycles(10); }

	if (timeout == 0) {
		uart_puts("  [FAIL] Echo device timeout\n");
		return 1;
	}

	// Step 2: Tell compute device to request data from echo device
	uart_puts("  Requesting peer data from compute device...\n");
	COMPUTE_PEER_OUT = 0x12345678;  // This triggers peer communication

	// Wait for peer data to arrive
	timeout = 10000;
	while ((COMPUTE_STATUS & STATUS_PEER_READY) == 0 && timeout-- > 0) { wait_cycles(100); }

	if (timeout == 0) {
		uart_puts("  [FAIL] Peer communication timeout\n");
		return 1;
	}

	// Step 3: Read the peer data from compute device
	uint32_t peer_data = COMPUTE_PEER_IN;
	uart_puts("  Received peer data: ");
	uart_puthex(peer_data);
	uart_puts("\n");

	// The peer data should be the echo device's current result
	uint32_t echo_result = ECHO_DATA_OUT;
	uart_puts("  Echo device result: ");
	uart_puthex(echo_result);
	uart_puts("\n");

	if (peer_data == echo_result) {
		uart_puts("  [PASS] Inter-device communication successful\n");
		return 0;
	} else {
		uart_puts("  [FAIL] Peer data mismatch\n");
		return 1;
	}
}

int main(int argc, char* argv[]) {
	(void)argc;
	(void)argv;

	uart_puts("\n");
	uart_puts("===========================================\n");
	uart_puts("  QEMU-SST Multi-Device Test\n");
	uart_puts("===========================================\n");
	uart_puts("\n");

	uart_puts("Device 1: Echo device at     ");
	uart_puthex(ECHO_DEVICE_BASE);
	uart_puts("\n");

	uart_puts("Device 2: Compute device at  ");
	uart_puthex(COMPUTE_DEVICE_BASE);
	uart_puts("\n");
	uart_puts("\n");

	int failures = 0;

	// Test 1: Echo device
	failures += test_echo_device();

	// Test 2: Compute device
	failures += test_compute_device();

	// Test 3: Inter-device communication
	failures += test_inter_device_communication();

	uart_puts("\n");
	uart_puts("===========================================\n");
	if (failures == 0) {
		uart_puts("  All tests PASSED!\n");
	} else {
		uart_puts("  Tests FAILED: ");
		uart_puthex(failures);
		uart_puts(" failures\n");
	}
	uart_puts("===========================================\n");
	uart_puts("\n");

	return failures;
}
