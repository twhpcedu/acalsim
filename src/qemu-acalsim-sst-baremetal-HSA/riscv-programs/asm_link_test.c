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
 * asm_link_test.c - Test C-Assembly Linkage
 *
 * Demonstrates calling RISC-V assembly functions from C main().
 * Tests the linkage between C code and assembly code.
 */

#include <stdint.h>

// Simple UART for output
#define UART_BASE 0x10000000
#define UART_TX   (*(volatile uint8_t *)UART_BASE)

// =============================================================================
// Assembly Function Prototypes
// =============================================================================

// Defined in asm_test.S
extern int asm_add(int a, int b);
extern int asm_multiply(int a, int b);
extern int asm_fibonacci(int n);
extern void asm_memory_copy(uint32_t *dest, const uint32_t *src, int count);
extern uint32_t asm_checksum(const uint32_t *data, int count);

// =============================================================================
// UART Helper Functions
// =============================================================================

void uart_putc(char c) {
    UART_TX = c;
}

void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

void uart_puthex(uint32_t val) {
    const char hex[] = "0123456789ABCDEF";
    uart_puts("0x");
    for (int i = 7; i >= 0; i--) {
        uart_putc(hex[(val >> (i * 4)) & 0xF]);
    }
}

void uart_putdec(int val) {
    if (val < 0) {
        uart_putc('-');
        val = -val;
    }

    char buf[11];
    int i = 10;
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

    while (1) {
        asm volatile("wfi");
    }
}

// =============================================================================
// Test Functions
// =============================================================================

int test_asm_add(void) {
    uart_puts("\n[TEST 1] asm_add\n");

    int result = asm_add(42, 58);
    uart_puts("  asm_add(42, 58) = ");
    uart_putdec(result);
    uart_puts("\n");

    if (result == 100) {
        uart_puts("  [PASS]\n");
        return 0;
    } else {
        uart_puts("  [FAIL] Expected 100\n");
        return 1;
    }
}

int test_asm_multiply(void) {
    uart_puts("\n[TEST 2] asm_multiply\n");

    int result = asm_multiply(12, 13);
    uart_puts("  asm_multiply(12, 13) = ");
    uart_putdec(result);
    uart_puts("\n");

    if (result == 156) {
        uart_puts("  [PASS]\n");
        return 0;
    } else {
        uart_puts("  [FAIL] Expected 156\n");
        return 1;
    }
}

int test_asm_fibonacci(void) {
    uart_puts("\n[TEST 3] asm_fibonacci\n");

    // Test fib(10) = 55
    int result = asm_fibonacci(10);
    uart_puts("  asm_fibonacci(10) = ");
    uart_putdec(result);
    uart_puts("\n");

    if (result == 55) {
        uart_puts("  [PASS]\n");
        return 0;
    } else {
        uart_puts("  [FAIL] Expected 55\n");
        return 1;
    }
}

int test_asm_memory_copy(void) {
    uart_puts("\n[TEST 4] asm_memory_copy\n");

    uint32_t src[5] = {0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555};
    uint32_t dest[5] = {0, 0, 0, 0, 0};

    uart_puts("  Copying 5 words from src to dest\n");
    asm_memory_copy(dest, src, 5);

    // Verify copy
    int success = 1;
    for (int i = 0; i < 5; i++) {
        if (dest[i] != src[i]) {
            success = 0;
            uart_puts("  [FAIL] Mismatch at index ");
            uart_putdec(i);
            uart_puts(": dest=");
            uart_puthex(dest[i]);
            uart_puts(" src=");
            uart_puthex(src[i]);
            uart_puts("\n");
        }
    }

    if (success) {
        uart_puts("  [PASS] All words copied correctly\n");
        return 0;
    } else {
        uart_puts("  [FAIL]\n");
        return 1;
    }
}

int test_asm_checksum(void) {
    uart_puts("\n[TEST 5] asm_checksum\n");

    uint32_t data[4] = {0x00000001, 0x00000002, 0x00000003, 0x00000004};
    uint32_t expected = 0x0000000A;  // 1 + 2 + 3 + 4 = 10

    uint32_t result = asm_checksum(data, 4);
    uart_puts("  asm_checksum([1,2,3,4]) = ");
    uart_puthex(result);
    uart_puts("\n");

    if (result == expected) {
        uart_puts("  [PASS]\n");
        return 0;
    } else {
        uart_puts("  [FAIL] Expected ");
        uart_puthex(expected);
        uart_puts("\n");
        return 1;
    }
}

// =============================================================================
// Main Function
// =============================================================================

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    uart_puts("===========================================\n");
    uart_puts("  C-Assembly Linkage Test\n");
    uart_puts("  Testing calls from C main() to RISC-V asm\n");
    uart_puts("===========================================\n");

    int failures = 0;

    failures += test_asm_add();
    failures += test_asm_multiply();
    failures += test_asm_fibonacci();
    failures += test_asm_memory_copy();
    failures += test_asm_checksum();

    uart_puts("\n===========================================\n");
    uart_puts("  Test Summary\n");
    uart_puts("===========================================\n");
    uart_puts("  Total failures: ");
    uart_putdec(failures);
    uart_puts("\n");

    if (failures == 0) {
        uart_puts("  [SUCCESS] All assembly tests passed!\n");
    } else {
        uart_puts("  [FAILURE] Some assembly tests failed!\n");
    }
    uart_puts("===========================================\n");

    return failures;
}
