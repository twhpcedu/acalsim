/*
 * Copyright 2023-2026 Playlab/ACAL
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
 * SST Device Test Program
 *
 * This RISC-V bare-metal program demonstrates communication with
 * SST devices using the UART-based protocol.
 */

#include "sst_device.h"

// Test data patterns
#define TEST_PATTERN_1 0xDEADBEEF
#define TEST_PATTERN_2 0xCAFEBABE
#define TEST_PATTERN_3 0x12345678
#define TEST_PATTERN_4 0xABCDEF00
#define TEST_PATTERN_5 0x55AA55AA

// Entry point (called from start.S)
void _start_c(void) {
	uart_puts("\r\n");
	uart_puts("========================================\r\n");
	uart_puts("QEMU-SST Integration Test\r\n");
	uart_puts("========================================\r\n");
	uart_puts("Testing SST device communication...\r\n");
	uart_puts("\r\n");

	// Test 1: Write to SST DATA_IN register
	uart_puts("Test 1: Write to SST device\r\n");
	uart_printf("  Writing 0x%X to DATA_IN...\r\n", TEST_PATTERN_1);

	if (sst_write(SST_DATA_IN, TEST_PATTERN_1)) {
		uart_puts("  [PASS] Write successful\r\n");
	} else {
		uart_puts("  [FAIL] Write failed\r\n");
		goto error;
	}
	uart_puts("\r\n");

	// Test 2: Poll device status
	uart_puts("Test 2: Poll device status\r\n");
	uart_puts("  Waiting for device ready...\r\n");

	if (sst_wait_ready()) {
		uart_puts("  [PASS] Device ready\r\n");
	} else {
		uart_puts("  [FAIL] Device timeout\r\n");
		goto error;
	}
	uart_puts("\r\n");

	// Test 3: Check data ready flag
	uart_puts("Test 3: Check data ready\r\n");

	if (sst_data_ready()) {
		uart_puts("  [PASS] Data ready flag set\r\n");
	} else {
		uart_puts("  [FAIL] Data not ready\r\n");
		goto error;
	}
	uart_puts("\r\n");

	// Test 4: Read from SST DATA_OUT register
	uart_puts("Test 4: Read from SST device\r\n");
	uint32_t result;

	if (sst_read(SST_DATA_OUT, &result)) {
		uart_printf("  Read value: 0x%X\r\n", result);
		if (result == TEST_PATTERN_1) {
			uart_puts("  [PASS] Echo test passed\r\n");
		} else {
			uart_printf("  [FAIL] Expected 0x%X, got 0x%X\r\n", TEST_PATTERN_1, result);
			goto error;
		}
	} else {
		uart_puts("  [FAIL] Read failed\r\n");
		goto error;
	}
	uart_puts("\r\n");

	// Test 5: Multiple transactions
	uart_puts("Test 5: Multiple transactions\r\n");
	uint32_t test_patterns[] = {TEST_PATTERN_2, TEST_PATTERN_3, TEST_PATTERN_4, TEST_PATTERN_5};
	int      num_tests       = sizeof(test_patterns) / sizeof(test_patterns[0]);
	int      passed          = 0;

	for (int i = 0; i < num_tests; i++) {
		uart_printf("  Transaction %d: 0x%X\r\n", i + 1, test_patterns[i]);

		// Write
		if (!sst_write(SST_DATA_IN, test_patterns[i])) {
			uart_puts("    Write failed\r\n");
			continue;
		}

		// Wait for ready
		if (!sst_wait_ready() || !sst_data_ready()) {
			uart_puts("    Device not ready\r\n");
			continue;
		}

		// Read back
		uint32_t read_val;
		if (!sst_read(SST_DATA_OUT, &read_val)) {
			uart_puts("    Read failed\r\n");
			continue;
		}

		// Verify
		if (read_val == test_patterns[i]) {
			uart_puts("    [PASS]\r\n");
			passed++;
		} else {
			uart_printf("    [FAIL] Expected 0x%X, got 0x%X\r\n", test_patterns[i], read_val);
		}
	}

	uart_printf("  Passed %d/%d transactions\r\n", passed, num_tests);
	if (passed == num_tests) {
		uart_puts("  [PASS] All transactions successful\r\n");
	} else {
		uart_puts("  [FAIL] Some transactions failed\r\n");
		goto error;
	}
	uart_puts("\r\n");

	// Success
	uart_puts("========================================\r\n");
	uart_puts("ALL TESTS PASSED!\r\n");
	uart_puts("========================================\r\n");
	uart_puts("\r\n");

	// Exit successfully
	volatile unsigned int* test_exit = (unsigned int*)0x100000;
	*test_exit                       = 0x5555;  // Success exit code

	while (1) { asm volatile("wfi"); }

error:
	uart_puts("\r\n");
	uart_puts("========================================\r\n");
	uart_puts("TEST FAILED!\r\n");
	uart_puts("========================================\r\n");
	uart_puts("\r\n");

	// Exit with error
	volatile unsigned int* test_exit_err = (unsigned int*)0x100000;
	*test_exit_err                       = 0x3333;  // Error exit code

	while (1) { asm volatile("wfi"); }
}

// Trap handler
void trap_handler_c(void) {
	uart_puts("\r\nTRAP! Entering infinite loop...\r\n");
	while (1) { asm volatile("wfi"); }
}
