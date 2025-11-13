/*
 * Multi-Device Test Program
 * Tests 4 SST devices with address-based routing
 */

#include <stdint.h>

// Device memory-mapped addresses
#define ECHO_DEVICE1_BASE    0x10200000
#define COMPUTE_DEVICE1_BASE 0x10300000
#define ECHO_DEVICE2_BASE    0x10400000
#define COMPUTE_DEVICE2_BASE 0x10500000

// Echo device registers (offset from base)
#define ECHO_DATA_IN  0x00
#define ECHO_DATA_OUT 0x04
#define ECHO_STATUS   0x08
#define ECHO_CONTROL  0x0C

// Compute device registers (offset from base)
#define COMPUTE_OPERAND_A 0x00
#define COMPUTE_OPERAND_B 0x04
#define COMPUTE_OPERATION 0x08
#define COMPUTE_RESULT    0x0C
#define COMPUTE_STATUS    0x10
#define COMPUTE_CONTROL   0x14

// Operation codes
#define OP_ADD 0
#define OP_SUB 1
#define OP_MUL 2
#define OP_DIV 3

// Helper macros
#define WRITE_REG(base, offset, value) (*((volatile uint32_t*)((base) + (offset))) = (value))

#define READ_REG(base, offset) (*((volatile uint32_t*)((base) + (offset))))

// UART for output
#define UART_BASE 0x10000000
#define UART_TX   0x00

void uart_putc(char c) { *((volatile uint32_t*)(UART_BASE + UART_TX)) = c; }

void uart_puts(const char* s) {
	while (*s) { uart_putc(*s++); }
}

void uart_puthex(uint32_t val) {
	const char hex[] = "0123456789ABCDEF";
	uart_puts("0x");
	for (int i = 7; i >= 0; i--) { uart_putc(hex[(val >> (i * 4)) & 0xF]); }
}

int main(void) {
	uint32_t result;
	int      pass_count = 0;
	int      fail_count = 0;

	uart_puts("\n========================================\n");
	uart_puts("Multi-Device Test - 4 Devices\n");
	uart_puts("========================================\n\n");

	// TEST 1: Echo Device 1 (0x10200000, latency 10)
	uart_puts("[TEST 1] Echo Device 1 @ 0x10200000\n");
	WRITE_REG(ECHO_DEVICE1_BASE, ECHO_DATA_IN, 0xDEADBEEF);
	result = READ_REG(ECHO_DEVICE1_BASE, ECHO_DATA_OUT);
	uart_puts("  Wrote: 0xDEADBEEF, Read: ");
	uart_puthex(result);
	if (result == 0xDEADBEEF) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// TEST 2: Compute Device 1 (0x10300000, latency 100) - Addition
	uart_puts("\n[TEST 2] Compute Device 1 @ 0x10300000 (ADD)\n");
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERAND_A, 42);
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERAND_B, 58);
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERATION, OP_ADD);
	result = READ_REG(COMPUTE_DEVICE1_BASE, COMPUTE_RESULT);
	uart_puts("  42 + 58 = ");
	uart_puthex(result);
	if (result == 100) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// TEST 3: Echo Device 2 (0x10400000, latency 5)
	uart_puts("\n[TEST 3] Echo Device 2 @ 0x10400000\n");
	WRITE_REG(ECHO_DEVICE2_BASE, ECHO_DATA_IN, 0xCAFEBABE);
	result = READ_REG(ECHO_DEVICE2_BASE, ECHO_DATA_OUT);
	uart_puts("  Wrote: 0xCAFEBABE, Read: ");
	uart_puthex(result);
	if (result == 0xCAFEBABE) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// TEST 4: Compute Device 2 (0x10500000, latency 50) - Multiplication
	uart_puts("\n[TEST 4] Compute Device 2 @ 0x10500000 (MUL)\n");
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERAND_A, 12);
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERAND_B, 5);
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERATION, OP_MUL);
	result = READ_REG(COMPUTE_DEVICE2_BASE, COMPUTE_RESULT);
	uart_puts("  12 * 5 = ");
	uart_puthex(result);
	if (result == 60) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// TEST 5: Interleaved access - alternate between devices
	uart_puts("\n[TEST 5] Interleaved Device Access\n");
	WRITE_REG(ECHO_DEVICE1_BASE, ECHO_DATA_IN, 0x11111111);
	WRITE_REG(ECHO_DEVICE2_BASE, ECHO_DATA_IN, 0x22222222);

	uint32_t r1 = READ_REG(ECHO_DEVICE1_BASE, ECHO_DATA_OUT);
	uint32_t r2 = READ_REG(ECHO_DEVICE2_BASE, ECHO_DATA_OUT);

	uart_puts("  Device1: ");
	uart_puthex(r1);
	uart_puts(", Device2: ");
	uart_puthex(r2);

	if (r1 == 0x11111111 && r2 == 0x22222222) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// TEST 6: All compute devices - simultaneous operations
	uart_puts("\n[TEST 6] Simultaneous Compute Operations\n");

	// Device 1: 100 / 4 = 25
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERAND_A, 100);
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERAND_B, 4);
	WRITE_REG(COMPUTE_DEVICE1_BASE, COMPUTE_OPERATION, OP_DIV);

	// Device 2: 50 - 20 = 30
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERAND_A, 50);
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERAND_B, 20);
	WRITE_REG(COMPUTE_DEVICE2_BASE, COMPUTE_OPERATION, OP_SUB);

	uint32_t res1 = READ_REG(COMPUTE_DEVICE1_BASE, COMPUTE_RESULT);
	uint32_t res2 = READ_REG(COMPUTE_DEVICE2_BASE, COMPUTE_RESULT);

	uart_puts("  Dev1 (100/4): ");
	uart_puthex(res1);
	uart_puts(", Dev2 (50-20): ");
	uart_puthex(res2);

	if (res1 == 25 && res2 == 30) {
		uart_puts(" [PASS]\n");
		pass_count++;
	} else {
		uart_puts(" [FAIL]\n");
		fail_count++;
	}

	// Summary
	uart_puts("\n========================================\n");
	uart_puts("Test Summary:\n");
	uart_puts("  Passed: ");
	uart_puthex(pass_count);
	uart_puts("\n  Failed: ");
	uart_puthex(fail_count);
	uart_puts("\n========================================\n");

	if (fail_count == 0) {
		uart_puts("All tests PASSED!\n");
	} else {
		uart_puts("Some tests FAILED!\n");
	}

	// Graceful exit
	return (fail_count == 0) ? 0 : 1;
}
