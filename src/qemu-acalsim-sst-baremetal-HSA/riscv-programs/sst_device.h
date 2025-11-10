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
 * SST Device Helper Library
 *
 * Provides helper functions for RISC-V bare-metal programs to communicate
 * with SST devices via the UART-based protocol.
 *
 * Protocol Format:
 *   Request:  SST:<CMD>:<ADDR>:<DATA>\n
 *   Response: SST:OK:<DATA>\n or SST:ERR:<CODE>\n
 */

#ifndef SST_DEVICE_H
#define SST_DEVICE_H

#include <stdint.h>
#include <stdbool.h>

// SST device base address
#define SST_DEVICE_BASE 0x20000000

// SST device registers
#define SST_DATA_IN  (SST_DEVICE_BASE + 0x00)
#define SST_DATA_OUT (SST_DEVICE_BASE + 0x04)
#define SST_STATUS   (SST_DEVICE_BASE + 0x08)

// Status bits
#define SST_STATUS_BUSY       (1 << 0)
#define SST_STATUS_DATA_READY (1 << 1)

// UART base for QEMU virt machine
#define UART_BASE 0x10000000
static volatile unsigned char *uart = (unsigned char *)UART_BASE;

// Forward declarations
void uart_putc(char c);
void uart_puts(const char *s);
void uart_printf(const char *format, ...);
char uart_getc(void);
void uart_gets(char *buffer, int size);
uint32_t parse_hex(const char *str);
void format_hex(char *buffer, uint32_t value);

/*
 * Write to SST device register
 *
 * @param addr Device register address
 * @param data Data to write
 * @return true if successful, false on error
 */
static inline bool sst_write(uint32_t addr, uint32_t data) {
    char buffer[64];

    // Send write command: SST:WRITE:ADDR:DATA\n
    uart_puts("SST:WRITE:");
    format_hex(buffer, addr);
    uart_puts(buffer);
    uart_putc(':');
    format_hex(buffer, data);
    uart_puts(buffer);
    uart_putc('\n');

    // Wait for response
    uart_gets(buffer, sizeof(buffer));

    // Check for SST:OK
    return (buffer[0] == 'S' && buffer[1] == 'S' && buffer[2] == 'T' &&
            buffer[3] == ':' && buffer[4] == 'O' && buffer[5] == 'K');
}

/*
 * Read from SST device register
 *
 * @param addr Device register address
 * @param data Pointer to store read data
 * @return true if successful, false on error
 */
static inline bool sst_read(uint32_t addr, uint32_t *data) {
    char buffer[64];

    // Send read command: SST:READ:ADDR:00000000\n
    uart_puts("SST:READ:");
    format_hex(buffer, addr);
    uart_puts(buffer);
    uart_puts(":00000000\n");

    // Wait for response
    uart_gets(buffer, sizeof(buffer));

    // Check for SST:OK:
    if (!(buffer[0] == 'S' && buffer[1] == 'S' && buffer[2] == 'T' &&
          buffer[3] == ':' && buffer[4] == 'O' && buffer[5] == 'K' &&
          buffer[6] == ':')) {
        return false;
    }

    // Parse hex data after SST:OK:
    *data = parse_hex(&buffer[7]);
    return true;
}

/*
 * Poll SST device status until not busy
 *
 * @return true if device ready, false on timeout
 */
static inline bool sst_wait_ready(void) {
    uint32_t status;
    int timeout = 10000;

    while (timeout-- > 0) {
        if (!sst_read(SST_STATUS, &status)) {
            return false;
        }

        if (!(status & SST_STATUS_BUSY)) {
            return true;
        }
    }

    return false;  // Timeout
}

/*
 * Check if SST device has data ready
 *
 * @return true if data ready, false otherwise
 */
static inline bool sst_data_ready(void) {
    uint32_t status;

    if (!sst_read(SST_STATUS, &status)) {
        return false;
    }

    return (status & SST_STATUS_DATA_READY) != 0;
}

#endif // SST_DEVICE_H
