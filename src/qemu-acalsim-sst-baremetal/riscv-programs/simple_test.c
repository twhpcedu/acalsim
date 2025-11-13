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
 * Simple RISC-V Bare-Metal Test Program
 *
 * This program demonstrates basic RISC-V execution in QEMU.
 * It performs simple operations and exits cleanly.
 */

// UART base address for QEMU virt machine
#define UART_BASE 0x10000000

// SST device base address (will be used in Phase 2B)
#define SST_DEVICE_BASE 0x20000000
#define SST_DATA_IN     ((volatile unsigned int*)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT    ((volatile unsigned int*)(SST_DEVICE_BASE + 0x04))
#define SST_STATUS      ((volatile unsigned int*)(SST_DEVICE_BASE + 0x08))

// Status bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DATA_READY (1 << 1)

// UART registers
static volatile unsigned char* uart = (unsigned char*)UART_BASE;

// Forward declarations
unsigned int test_function(unsigned int x, unsigned int y);

// Simple UART output function
void uart_putc(char c) { *uart = c; }

void uart_puts(const char* s) {
	while (*s) { uart_putc(*s++); }
}

void uart_put_hex(unsigned int val) {
	const char hex[] = "0123456789ABCDEF";
	uart_puts("0x");
	for (int i = 28; i >= 0; i -= 4) { uart_putc(hex[(val >> i) & 0xF]); }
}

// Entry point (called from start.S)
void _start_c(void) {
	uart_puts("\r\n");
	uart_puts("================================\r\n");
	uart_puts("RISC-V Bare-Metal Test Program\r\n");
	uart_puts("================================\r\n");
	uart_puts("Running in QEMU...\r\n");
	uart_puts("\r\n");

	// Perform some simple arithmetic to prove we're executing
	unsigned int a = 0xDEADBEEF;
	unsigned int b = 0xCAFEBABE;
	unsigned int c = a + b;

	uart_puts("Test 1: Simple arithmetic\r\n");
	uart_puts("  a = ");
	uart_put_hex(a);
	uart_puts("\r\n");
	uart_puts("  b = ");
	uart_put_hex(b);
	uart_puts("\r\n");
	uart_puts("  c = a + b = ");
	uart_put_hex(c);
	uart_puts("\r\n");
	uart_puts("\r\n");

	// Test 2: Loop counter
	uart_puts("Test 2: Loop execution\r\n");
	for (int i = 0; i < 5; i++) {
		uart_puts("  Iteration ");
		uart_putc('0' + i);
		uart_puts("\r\n");
	}
	uart_puts("\r\n");

	// Test 3: Function call
	uart_puts("Test 3: Function call\r\n");
	unsigned int result = test_function(10, 20);
	uart_puts("  test_function(10, 20) = ");
	uart_put_hex(result);
	uart_puts("\r\n");
	uart_puts("\r\n");

	// Success message
	uart_puts("================================\r\n");
	uart_puts("All tests completed successfully!\r\n");
	uart_puts("================================\r\n");
	uart_puts("\r\n");

	// Exit - write to QEMU's test device
	// Address 0x100000 is the test device exit address in QEMU virt machine
	volatile unsigned int* test_exit = (unsigned int*)0x100000;
	*test_exit                       = 0x5555;  // Exit with success code

	// If exit doesn't work, loop forever
	while (1) {
		asm volatile("wfi");  // Wait for interrupt
	}
}

// Test function to demonstrate function calls
unsigned int test_function(unsigned int x, unsigned int y) { return x * 2 + y * 3; }

// Handle any exceptions/interrupts (just loop)
void trap_handler_c(void) {
	uart_puts("\r\nTRAP! Entering infinite loop...\r\n");
	while (1) { asm volatile("wfi"); }
}
