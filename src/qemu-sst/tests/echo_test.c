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
 * @file echo_test.c
 * @brief Echo device test program for RISC-V
 *
 * This is a reference implementation showing how a real RISC-V test program
 * would interact with the memory-mapped echo device. In the current SST
 * simulation, the QEMU component simulates this behavior.
 *
 * For full QEMU integration, this program would be compiled for RISC-V and
 * run inside QEMU, with the device operations forwarded to the ACALSim device
 * via SST links.
 */

#include <stdint.h>

// Device base address (memory-mapped)
#define DEVICE_BASE 0x10000000

// Device register offsets
#define REG_DATA_IN  (DEVICE_BASE + 0x00)  // Write-only
#define REG_DATA_OUT (DEVICE_BASE + 0x04)  // Read-only
#define REG_STATUS   (DEVICE_BASE + 0x08)  // Read-only
#define REG_CONTROL  (DEVICE_BASE + 0x0C)  // Read/Write

// Status register bits
#define STATUS_BUSY       0x01  // Device is processing
#define STATUS_DATA_READY 0x02  // Data is ready in DATA_OUT

// Control register bits
#define CONTROL_RESET 0x01  // Reset device

// Test configuration
#define TEST_PATTERN   0xDEADBEEF
#define NUM_ITERATIONS 5
#define MAX_WAIT_CYCLES 1000

// Simple print function (would be replaced with UART output in real QEMU)
void print_str(const char* str);
void print_hex(uint32_t val);
void print_result(const char* msg, int passed);

/**
 * @brief Write to device register
 */
static inline void write_reg(volatile uint32_t* addr, uint32_t value) {
	*addr = value;
}

/**
 * @brief Read from device register
 */
static inline uint32_t read_reg(volatile uint32_t* addr) {
	return *addr;
}

/**
 * @brief Wait for device to complete operation
 * @return 0 on success, -1 on timeout
 */
int wait_device_ready(void) {
	volatile uint32_t* status = (volatile uint32_t*)REG_STATUS;
	int wait_cycles = 0;

	// Wait for device to clear BUSY and set DATA_READY
	while (wait_cycles < MAX_WAIT_CYCLES) {
		uint32_t status_val = read_reg(status);

		// Check if data is ready
		if (status_val & STATUS_DATA_READY) {
			return 0;  // Success
		}

		wait_cycles++;
	}

	return -1;  // Timeout
}

/**
 * @brief Reset device to initial state
 */
void reset_device(void) {
	volatile uint32_t* control = (volatile uint32_t*)REG_CONTROL;
	write_reg(control, CONTROL_RESET);
}

/**
 * @brief Test echo device with single pattern
 * @param pattern Test pattern to write
 * @return 0 on success, -1 on failure
 */
int test_echo(uint32_t pattern) {
	volatile uint32_t* data_in  = (volatile uint32_t*)REG_DATA_IN;
	volatile uint32_t* data_out = (volatile uint32_t*)REG_DATA_OUT;
	uint32_t result;

	// Write test pattern to device
	write_reg(data_in, pattern);

	// Wait for device to process
	if (wait_device_ready() != 0) {
		print_str("ERROR: Device timeout\n");
		return -1;
	}

	// Read echoed data
	result = read_reg(data_out);

	// Verify result
	if (result != pattern) {
		print_str("ERROR: Data mismatch - expected ");
		print_hex(pattern);
		print_str(", got ");
		print_hex(result);
		print_str("\n");
		return -1;
	}

	return 0;
}

/**
 * @brief Main test program
 */
int main(void) {
	int i;
	int failures = 0;
	int successes = 0;

	print_str("==============================================\n");
	print_str("Echo Device Test Program\n");
	print_str("==============================================\n\n");

	// Reset device before starting
	print_str("Resetting device...\n");
	reset_device();

	// Run multiple test iterations
	for (i = 0; i < NUM_ITERATIONS; i++) {
		uint32_t pattern = TEST_PATTERN + i;

		print_str("Iteration ");
		print_hex(i + 1);
		print_str(": Testing pattern ");
		print_hex(pattern);
		print_str("... ");

		if (test_echo(pattern) == 0) {
			print_str("PASSED\n");
			successes++;
		} else {
			print_str("FAILED\n");
			failures++;
		}
	}

	// Print summary
	print_str("\n==============================================\n");
	print_str("Test Summary:\n");
	print_str("  Total:     ");
	print_hex(NUM_ITERATIONS);
	print_str("\n  Successes: ");
	print_hex(successes);
	print_str("\n  Failures:  ");
	print_hex(failures);
	print_str("\n==============================================\n");

	if (failures == 0) {
		print_str("\n*** ALL TESTS PASSED ***\n");
		return 0;
	} else {
		print_str("\n*** SOME TESTS FAILED ***\n");
		return 1;
	}
}

// ==============================================================================
// Simple Print Functions (placeholder implementations)
// ==============================================================================

/**
 * @brief Print string (stub - would use UART in real implementation)
 */
void print_str(const char* str) {
	// In real QEMU, this would write to UART
	// For SST simulation, the QEMU component logs events
	(void)str;  // Suppress unused warning
}

/**
 * @brief Print hex value (stub - would use UART in real implementation)
 */
void print_hex(uint32_t val) {
	// In real QEMU, this would format and write to UART
	(void)val;  // Suppress unused warning
}

/**
 * @brief Print test result (stub)
 */
void print_result(const char* msg, int passed) {
	print_str(msg);
	print_str(": ");
	if (passed) {
		print_str("PASSED\n");
	} else {
		print_str("FAILED\n");
	}
}
