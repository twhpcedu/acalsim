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

#include "sst_device.h"

// UART output functions
void uart_putc(char c) {
    *uart = c;
}

void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

// Simple implementation of uart_printf (supports %d, %x, %s only)
void uart_printf(const char *format, ...) {
    const char *p = format;
    unsigned int *args = (unsigned int *)&format + 1;
    int arg_index = 0;

    while (*p) {
        if (*p == '%' && *(p + 1)) {
            p++;
            switch (*p) {
            case 'd': {  // Decimal integer
                int val = args[arg_index++];
                if (val < 0) {
                    uart_putc('-');
                    val = -val;
                }
                char buffer[12];
                int i = 0;
                do {
                    buffer[i++] = '0' + (val % 10);
                    val /= 10;
                } while (val > 0);
                while (i > 0) {
                    uart_putc(buffer[--i]);
                }
                break;
            }
            case 'x':
            case 'X': {  // Hexadecimal
                unsigned int val = args[arg_index++];
                char buffer[9];
                format_hex(buffer, val);
                uart_puts(buffer);
                break;
            }
            case 's': {  // String
                const char *str = (const char *)args[arg_index++];
                uart_puts(str);
                break;
            }
            case 'c': {  // Character
                char c = (char)args[arg_index++];
                uart_putc(c);
                break;
            }
            default:
                uart_putc(*p);
                break;
            }
        } else {
            uart_putc(*p);
        }
        p++;
    }
}

// UART input functions (note: blocking, no interrupts)
char uart_getc(void) {
    // In real implementation, would check UART status
    // For simulation, just read directly
    return *uart;
}

void uart_gets(char *buffer, int size) {
    int i = 0;
    while (i < size - 1) {
        char c = uart_getc();
        if (c == '\n' || c == '\r' || c == '\0') {
            break;
        }
        buffer[i++] = c;
    }
    buffer[i] = '\0';
}

// Parse hexadecimal string to uint32_t
uint32_t parse_hex(const char *str) {
    uint32_t value = 0;
    while (*str) {
        char c = *str++;
        if (c >= '0' && c <= '9') {
            value = (value << 4) | (c - '0');
        } else if (c >= 'A' && c <= 'F') {
            value = (value << 4) | (c - 'A' + 10);
        } else if (c >= 'a' && c <= 'f') {
            value = (value << 4) | (c - 'a' + 10);
        } else {
            break;  // Stop on non-hex character
        }
    }
    return value;
}

// Format uint32_t as 8-character hexadecimal string
void format_hex(char *buffer, uint32_t value) {
    const char hex[] = "0123456789ABCDEF";
    for (int i = 7; i >= 0; i--) {
        buffer[i] = hex[value & 0xF];
        value >>= 4;
    }
    buffer[8] = '\0';
}
