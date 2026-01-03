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
 * asm_test.S - RISC-V Assembly Test Functions
 *
 * Demonstrates calling assembly functions from C main().
 */

.section .text

# =============================================================================
# asm_add - Add two 32-bit integers
# =============================================================================
# int asm_add(int a, int b)
# Arguments: a0 = a, a1 = b
# Returns: a0 = a + b

.global asm_add
.type asm_add, @function
asm_add:
    add a0, a0, a1      # a0 = a0 + a1
    ret                  # Return

# =============================================================================
# asm_multiply - Multiply two 32-bit integers
# =============================================================================
# int asm_multiply(int a, int b)
# Arguments: a0 = a, a1 = b
# Returns: a0 = a * b

.global asm_multiply
.type asm_multiply, @function
asm_multiply:
    mul a0, a0, a1      # a0 = a0 * a1
    ret                  # Return

# =============================================================================
# asm_fibonacci - Calculate Fibonacci number recursively
# =============================================================================
# int asm_fibonacci(int n)
# Arguments: a0 = n
# Returns: a0 = fib(n)

.global asm_fibonacci
.type asm_fibonacci, @function
asm_fibonacci:
    # Base cases: fib(0) = 0, fib(1) = 1
    li t0, 2
    bltu a0, t0, fib_base_case

    # Save registers
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)

    # Save n
    mv s0, a0

    # Calculate fib(n-1)
    addi a0, s0, -1
    call asm_fibonacci
    mv s1, a0           # s1 = fib(n-1)

    # Calculate fib(n-2)
    addi a0, s0, -2
    call asm_fibonacci
    add a0, a0, s1      # a0 = fib(n-2) + fib(n-1)

    # Restore registers
    lw s1, 4(sp)
    lw s0, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    ret

fib_base_case:
    # fib(0) = 0, fib(1) = 1
    ret

# =============================================================================
# asm_memory_copy - Copy memory from src to dest
# =============================================================================
# void asm_memory_copy(uint32_t *dest, const uint32_t *src, int count)
# Arguments: a0 = dest, a1 = src, a2 = count
# Returns: nothing

.global asm_memory_copy
.type asm_memory_copy, @function
asm_memory_copy:
    beqz a2, copy_done  # If count == 0, done

copy_loop:
    lw t0, 0(a1)        # Load from src
    sw t0, 0(a0)        # Store to dest
    addi a0, a0, 4      # dest++
    addi a1, a1, 4      # src++
    addi a2, a2, -1     # count--
    bnez a2, copy_loop  # Continue if count != 0

copy_done:
    ret

# =============================================================================
# asm_checksum - Calculate simple checksum of array
# =============================================================================
# uint32_t asm_checksum(const uint32_t *data, int count)
# Arguments: a0 = data, a1 = count
# Returns: a0 = checksum

.global asm_checksum
.type asm_checksum, @function
asm_checksum:
    li a2, 0            # checksum = 0

    beqz a1, checksum_done

checksum_loop:
    lw t0, 0(a0)        # Load word
    add a2, a2, t0      # checksum += word
    addi a0, a0, 4      # data++
    addi a1, a1, -1     # count--
    bnez a1, checksum_loop

checksum_done:
    mv a0, a2           # Return checksum
    ret
