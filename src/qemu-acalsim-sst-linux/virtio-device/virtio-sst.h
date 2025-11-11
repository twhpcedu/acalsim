/*
 * VirtIO SST Device for QEMU - Header
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
 * VirtIO SST Device
 *
 * This device implements a VirtIO interface for SST (Structural Simulation
 * Toolkit) communication in Linux guests. It bridges Linux kernel drivers
 * with SST simulator components via Unix sockets.
 *
 * Architecture:
 * - VirtIO device in QEMU (this file)
 * - Unix socket connection to SST
 * - Linux kernel driver (virtio-sst.ko)
 * - User-space applications via /dev/sst*
 *
 * Features:
 * - Standard VirtIO queues (request/response/event)
 * - Binary protocol compatible with existing SST components
 * - Async event notifications
 * - Multi-queue support (future)
 * - Compatible with VirtIO 1.0+ specification
 */

#ifndef HW_VIRTIO_SST_H
#define HW_VIRTIO_SST_H

#include "qemu/osdep.h"
#include "hw/virtio/virtio.h"
#include "qom/object.h"
#include "sst-protocol.h"

#define TYPE_VIRTIO_SST "virtio-sst-device"
OBJECT_DECLARE_SIMPLE_TYPE(VirtIOSST, VIRTIO_SST)

/*
 * VirtQueue Configuration
 */
#define VIRTIO_SST_QUEUE_SIZE   128    // Queue depth

/*
 * Request Tracking
 */
typedef struct SSTRequestInfo {
    uint64_t request_id;
    VirtQueueElement *elem;
    struct SSTRequest *req;
    struct SSTResponse *resp;
    QLIST_ENTRY(SSTRequestInfo) next;
} SSTRequestInfo;

/*
 * VirtIO SST Device State
 */
struct VirtIOSST {
    VirtIODevice parent_obj;

    /* VirtQueues */
    VirtQueue *req_vq;          // Request queue (driver -> device)
    VirtQueue *resp_vq;         // Response queue (device -> driver)
    VirtQueue *event_vq;        // Event queue (device -> driver, async)

    /* Configuration Space */
    struct SSTConfig config;

    /* SST Socket Connection */
    char *socket_path;          // Unix socket path to SST
    int socket_fd;              // Socket file descriptor
    bool connected;             // Connection status
    QIOChannel *socket_channel; // QEMU I/O channel for async
    GSource *socket_watch;      // Socket watch for async events

    /* Device Properties */
    uint32_t device_id;         // SST device ID
    uint64_t features;          // Enabled features

    /* Request Tracking */
    uint64_t next_request_id;   // Next request ID
    QLIST_HEAD(, SSTRequestInfo) pending_requests;  // Pending request list

    /* Statistics */
    uint64_t total_requests;
    uint64_t total_responses;
    uint64_t total_events;
    uint64_t total_errors;
};

/*
 * Function Prototypes
 */

/* Device Lifecycle */
void virtio_sst_realize(DeviceState *dev, Error **errp);
void virtio_sst_unrealize(DeviceState *dev);
void virtio_sst_reset(VirtIODevice *vdev);

/* VirtQueue Handlers */
void virtio_sst_handle_request(VirtIODevice *vdev, VirtQueue *vq);
void virtio_sst_handle_response(VirtIODevice *vdev, VirtQueue *vq);
void virtio_sst_handle_event(VirtIODevice *vdev, VirtQueue *vq);

/* SST Communication */
bool virtio_sst_connect(VirtIOSST *s, Error **errp);
void virtio_sst_disconnect(VirtIOSST *s);
bool virtio_sst_send_request(VirtIOSST *s, struct SSTRequest *req,
                             struct SSTResponse *resp);

/* Request Processing */
void virtio_sst_process_request(VirtIOSST *s, VirtQueueElement *elem);
void virtio_sst_complete_request(VirtIOSST *s, SSTRequestInfo *info);

/* Event Handling */
void virtio_sst_inject_event(VirtIOSST *s, struct SSTEvent *event);

/* Feature Negotiation */
uint64_t virtio_sst_get_features(VirtIODevice *vdev, uint64_t requested_features,
                                Error **errp);
void virtio_sst_set_features(VirtIODevice *vdev, uint64_t features);

/* Configuration */
void virtio_sst_get_config(VirtIODevice *vdev, uint8_t *config);
void virtio_sst_set_config(VirtIODevice *vdev, const uint8_t *config);

/* Status */
void virtio_sst_set_status(VirtIODevice *vdev, uint8_t status);

#endif /* HW_VIRTIO_SST_H */
