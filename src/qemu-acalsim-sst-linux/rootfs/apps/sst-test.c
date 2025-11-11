/*
 * SST Device Test Application
 *
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
 * Basic test application for SST device
 *
 * Usage:
 *   ./sst-test [device]
 *
 * Example:
 *   ./sst-test /dev/sst0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include "../../virtio-device/sst-protocol.h"

#define DEFAULT_DEVICE "/dev/sst0"

/*
 * Test: NOOP Request
 */
int test_noop(int fd)
{
    struct SSTRequest req = {0};
    struct SSTResponse resp = {0};
    int ret;

    printf("\n[TEST] NOOP Request\n");
    printf("  Sending NOOP request...\n");

    req.type = SST_REQ_NOOP;
    req.user_data = 0x1234;

    ret = write(fd, &req, sizeof(req));
    if (ret != sizeof(req)) {
        perror("  Failed to write request");
        return -1;
    }

    ret = read(fd, &resp, sizeof(resp));
    if (ret != sizeof(resp)) {
        perror("  Failed to read response");
        return -1;
    }

    printf("  Status: %s\n", sst_status_str(resp.status));
    printf("  User data: 0x%lx\n", resp.user_data);

    return (resp.status == SST_STATUS_OK) ? 0 : -1;
}

/*
 * Test: ECHO Request
 */
int test_echo(int fd)
{
    struct SSTRequest req = {0};
    struct SSTResponse resp = {0};
    const char *test_data = "Hello SST!";
    int ret;

    printf("\n[TEST] ECHO Request\n");
    printf("  Sending ECHO request: \"%s\"\n", test_data);

    req.type = SST_REQ_ECHO;
    req.user_data = 0x5678;
    strncpy((char *)req.payload.data, test_data, SST_MAX_DATA_SIZE - 1);

    ret = write(fd, &req, sizeof(req));
    if (ret != sizeof(req)) {
        perror("  Failed to write request");
        return -1;
    }

    ret = read(fd, &resp, sizeof(resp));
    if (ret != sizeof(resp)) {
        perror("  Failed to read response");
        return -1;
    }

    printf("  Status: %s\n", sst_status_str(resp.status));
    printf("  Echo data: \"%s\"\n", (char *)resp.payload.data);

    if (strcmp((char *)resp.payload.data, test_data) != 0) {
        printf("  ERROR: Echo data mismatch!\n");
        return -1;
    }

    return 0;
}

/*
 * Test: GET_INFO Request
 */
int test_get_info(int fd)
{
    struct SSTRequest req = {0};
    struct SSTResponse resp = {0};
    int ret;

    printf("\n[TEST] GET_INFO Request\n");
    printf("  Querying device information...\n");

    req.type = SST_REQ_GET_INFO;

    ret = write(fd, &req, sizeof(req));
    if (ret != sizeof(req)) {
        perror("  Failed to write request");
        return -1;
    }

    ret = read(fd, &resp, sizeof(resp));
    if (ret != sizeof(resp)) {
        perror("  Failed to read response");
        return -1;
    }

    printf("  Status: %s\n", sst_status_str(resp.status));
    printf("  Device Info:\n");
    printf("    Version: 0x%08x\n", resp.payload.info.version);
    printf("    Capabilities: 0x%08x\n", resp.payload.info.capabilities);
    printf("    Max Compute Units: %lu\n", resp.payload.info.max_compute_units);
    printf("    Memory Size: %lu MB\n", resp.payload.info.memory_size / (1024*1024));

    return (resp.status == SST_STATUS_OK) ? 0 : -1;
}

/*
 * Test: COMPUTE Request
 */
int test_compute(int fd)
{
    struct SSTRequest req = {0};
    struct SSTResponse resp = {0};
    struct timespec start, end;
    double elapsed_ms;
    int ret;

    printf("\n[TEST] COMPUTE Request\n");
    printf("  Submitting compute request (1000 units)...\n");

    req.type = SST_REQ_COMPUTE;
    req.user_data = 0xABCD;
    req.payload.compute.compute_units = 1000;
    req.payload.compute.latency_model = 0;

    clock_gettime(CLOCK_MONOTONIC, &start);

    ret = write(fd, &req, sizeof(req));
    if (ret != sizeof(req)) {
        perror("  Failed to write request");
        return -1;
    }

    ret = read(fd, &resp, sizeof(resp));
    if (ret != sizeof(resp)) {
        perror("  Failed to read response");
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                 (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("  Status: %s\n", sst_status_str(resp.status));
    printf("  Simulated Cycles: %lu\n", resp.payload.compute.cycles);
    printf("  Timestamp: %lu\n", resp.payload.compute.timestamp);
    printf("  Wall-clock time: %.2f ms\n", elapsed_ms);

    return (resp.status == SST_STATUS_OK) ? 0 : -1;
}

/*
 * Test: Multiple Sequential Requests
 */
int test_sequential(int fd, int count)
{
    struct SSTRequest req = {0};
    struct SSTResponse resp = {0};
    struct timespec start, end;
    double elapsed_ms;
    int i, ret;

    printf("\n[TEST] Sequential Requests (%d iterations)\n", count);

    req.type = SST_REQ_COMPUTE;
    req.payload.compute.compute_units = 100;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (i = 0; i < count; i++) {
        req.user_data = i;

        ret = write(fd, &req, sizeof(req));
        if (ret != sizeof(req)) {
            perror("  Failed to write request");
            return -1;
        }

        ret = read(fd, &resp, sizeof(resp));
        if (ret != sizeof(resp)) {
            perror("  Failed to read response");
            return -1;
        }

        if (resp.status != SST_STATUS_OK) {
            printf("  Request %d failed: %s\n", i, sst_status_str(resp.status));
            return -1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                 (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("  Completed %d requests in %.2f ms\n", count, elapsed_ms);
    printf("  Average: %.2f ms/request\n", elapsed_ms / count);

    return 0;
}

/*
 * Main Program
 */
int main(int argc, char *argv[])
{
    const char *device = DEFAULT_DEVICE;
    int fd;
    int failed = 0;

    printf("============================================\n");
    printf("  SST Device Test Application\n");
    printf("============================================\n");

    if (argc > 1) {
        device = argv[1];
    }

    printf("\nOpening device: %s\n", device);

    fd = open(device, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        printf("\nMake sure:\n");
        printf("  1. virtio-sst kernel module is loaded\n");
        printf("  2. QEMU has virtio-sst-device configured\n");
        printf("  3. SST simulator is running\n");
        return 1;
    }

    printf("Device opened successfully (fd=%d)\n", fd);

    /* Run tests */
    if (test_noop(fd) != 0) {
        printf("  FAILED\n");
        failed++;
    } else {
        printf("  PASSED\n");
    }

    if (test_echo(fd) != 0) {
        printf("  FAILED\n");
        failed++;
    } else {
        printf("  PASSED\n");
    }

    if (test_get_info(fd) != 0) {
        printf("  FAILED\n");
        failed++;
    } else {
        printf("  PASSED\n");
    }

    if (test_compute(fd) != 0) {
        printf("  FAILED\n");
        failed++;
    } else {
        printf("  PASSED\n");
    }

    if (test_sequential(fd, 10) != 0) {
        printf("  FAILED\n");
        failed++;
    } else {
        printf("  PASSED\n");
    }

    /* Close device */
    close(fd);

    /* Summary */
    printf("\n============================================\n");
    printf("  Test Summary\n");
    printf("============================================\n");
    printf("  Tests run: 5\n");
    printf("  Passed: %d\n", 5 - failed);
    printf("  Failed: %d\n", failed);
    printf("============================================\n");

    return (failed > 0) ? 1 : 0;
}
