/**
 * HSA Multi-Accelerator Demo
 *
 * Linux application demonstrating parallel execution across multiple
 * AI accelerators using HSA-style programming model via /dev/sst0.
 *
 * Copyright 2023-2026 Playlab/ACAL
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Include SST protocol (relative to build location)
#include "sst-protocol.h"

#define NUM_DEVICES     4
#define MATRIX_SIZE     1024
#define ROWS_PER_DEVICE (MATRIX_SIZE / NUM_DEVICES)

// Work descriptor for each device
typedef struct {
	int      device_id;
	int      fd;  // File descriptor to /dev/sst0
	int      start_row;
	int      end_row;
	uint64_t cycles;  // Simulated cycles for this device
	int      status;  // Result status
} work_t;

// Global device file descriptor
static int g_sst_fd = -1;

/**
 * Worker thread: compute on one accelerator device
 */
void* compute_kernel(void* arg) {
	work_t*            w = (work_t*)arg;
	struct SSTRequest  req;
	struct SSTResponse resp;

	memset(&req, 0, sizeof(req));
	req.type       = SST_REQ_COMPUTE;
	req.request_id = w->device_id + 1;  // Unique ID per device
	req.user_data  = (uint64_t)w;

	// Compute units = number of rows to process
	req.payload.compute.compute_units = w->end_row - w->start_row;
	req.payload.compute.latency_model = 0;  // Default model

	printf("  [Thread %d] Submitting to device %d (rows %d-%d, %lu units)...", w->device_id, w->device_id, w->start_row,
	       w->end_row - 1, req.payload.compute.compute_units);
	fflush(stdout);

	// Submit request
	if (write(w->fd, &req, sizeof(req)) != sizeof(req)) {
		fprintf(stderr, "\n  [Thread %d] write() failed: %s\n", w->device_id, strerror(errno));
		w->status = -1;
		return NULL;
	}

	// Wait for response
	if (read(w->fd, &resp, sizeof(resp)) != sizeof(resp)) {
		fprintf(stderr, "\n  [Thread %d] read() failed: %s\n", w->device_id, strerror(errno));
		w->status = -1;
		return NULL;
	}

	// Check status
	if (resp.status != SST_STATUS_OK) {
		fprintf(stderr, "\n  [Thread %d] SST error: %s\n", w->device_id, sst_status_str(resp.status));
		w->status = -1;
		return NULL;
	}

	w->cycles = resp.payload.compute.cycles;
	w->status = 0;

	printf(" OK (%lu cycles)\n", w->cycles);
	return NULL;
}

/**
 * Main demo
 */
int main() {
	printf("============================================\n");
	printf("  HSA Multi-Accelerator Demo\n");
	printf("  Matrix Multiplication Benchmark\n");
	printf("============================================\n\n");

	// Initialize HSA Runtime
	printf("Initializing HSA Runtime...\n");
	printf("  Opening device: /dev/sst0\n");

	g_sst_fd = open("/dev/sst0", O_RDWR);
	if (g_sst_fd < 0) {
		perror("  Failed to open /dev/sst0");
		printf("\nError: Make sure:\n");
		printf("  1. Kernel driver loaded: insmod /virtio-sst.ko\n");
		printf("  2. Device exists: ls -l /dev/sst0\n");
		printf("  3. SST simulator is running\n");
		return 1;
	}

	printf("  Device opened successfully (fd=%d)\n\n", g_sst_fd);

	// Query device information
	printf("Discovering Accelerators...\n");

	struct SSTRequest  req;
	struct SSTResponse resp;

	memset(&req, 0, sizeof(req));
	req.type       = SST_REQ_GET_INFO;
	req.request_id = 0;

	if (write(g_sst_fd, &req, sizeof(req)) != sizeof(req) || read(g_sst_fd, &resp, sizeof(resp)) != sizeof(resp)) {
		fprintf(stderr, "  Failed to query device info\n");
		close(g_sst_fd);
		return 1;
	}

	if (resp.status != SST_STATUS_OK) {
		fprintf(stderr, "  Device info query failed: %s\n", sst_status_str(resp.status));
		close(g_sst_fd);
		return 1;
	}

	printf("  Found %d HSA compute devices:\n", NUM_DEVICES);
	for (int i = 0; i < NUM_DEVICES; i++) {
		printf("    Device %d: %lu compute units\n", i, resp.payload.info.max_compute_units / NUM_DEVICES);
	}
	printf("  Total compute power: %lu units\n\n", resp.payload.info.max_compute_units);

	// Prepare workload
	printf("Preparing Workload...\n");
	printf("  Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
	printf("  Total elements: %d\n", MATRIX_SIZE * MATRIX_SIZE);
	printf("  Elements per device: %d\n", (MATRIX_SIZE * MATRIX_SIZE) / NUM_DEVICES);
	printf("  Work distribution:\n");

	work_t    work[NUM_DEVICES];
	pthread_t threads[NUM_DEVICES];

	for (int i = 0; i < NUM_DEVICES; i++) {
		work[i].device_id = i;
		work[i].start_row = i * ROWS_PER_DEVICE;
		work[i].end_row   = (i + 1) * ROWS_PER_DEVICE;
		work[i].cycles    = 0;
		work[i].status    = 0;

		// Each thread gets its own file descriptor for concurrent access
		work[i].fd = open("/dev/sst0", O_RDWR);
		if (work[i].fd < 0) {
			perror("  Failed to open device for thread");
			close(g_sst_fd);
			return 1;
		}

		printf("    Device %d: rows %d-%d\n", i, work[i].start_row, work[i].end_row - 1);
	}
	printf("\n");

	// Submit kernels in parallel
	printf("Submitting Kernels...\n");
	for (int i = 0; i < NUM_DEVICES; i++) {
		if (pthread_create(&threads[i], NULL, compute_kernel, &work[i]) != 0) {
			fprintf(stderr, "  Failed to create thread %d\n", i);
			close(g_sst_fd);
			return 1;
		}
	}

	// Wait for completion
	printf("\nWaiting for Completion...\n");
	for (int i = 0; i < NUM_DEVICES; i++) {
		pthread_join(threads[i], NULL);
		if (work[i].status == 0) {
			printf("  [Thread %d] Device %d completed in %lu cycles\n", i, i, work[i].cycles);
		} else {
			printf("  [Thread %d] Device %d FAILED\n", i, i);
		}
	}
	printf("\n");

	// Aggregate results
	printf("Aggregating Results...\n");
	uint64_t max_cycles = 0;
	int      failed     = 0;

	for (int i = 0; i < NUM_DEVICES; i++) {
		if (work[i].status != 0) {
			failed++;
		} else if (work[i].cycles > max_cycles) {
			max_cycles = work[i].cycles;
		}
	}

	if (failed > 0) {
		printf("  ERROR: %d devices failed\n", failed);
		close(g_sst_fd);
		return 1;
	}

	printf("  Total simulated cycles: %lu\n", max_cycles);
	printf("  Effective parallelism: %dx\n", NUM_DEVICES);

	// Calculate performance
	uint64_t total_ops     = (uint64_t)MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
	double   ops_per_cycle = (double)total_ops / max_cycles;

	printf("\nPerformance Summary:\n");
	printf("  Total operations: %lu (matrix multiply)\n", total_ops);
	printf("  Simulated cycles: %lu\n", max_cycles);
	printf("  Operations/cycle: %.1f\n", ops_per_cycle);

	// Cleanup
	for (int i = 0; i < NUM_DEVICES; i++) { close(work[i].fd); }
	close(g_sst_fd);

	printf("\n============================================\n");
	printf("Demo completed successfully!\n");
	printf("============================================\n");

	return 0;
}
