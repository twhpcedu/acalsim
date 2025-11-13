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
 * Linux Kernel Driver for VirtIO SST Device
 *
 * This driver creates /dev/sst* character devices for user-space
 * applications to communicate with SST simulator via VirtIO.
 */

#ifndef SST_DEVICE_H
#define SST_DEVICE_H

#include <linux/cdev.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/wait.h>

/* Import protocol definitions */
#include "../virtio-device/sst-protocol.h"

/*
 * Driver Constants
 */
#define SST_DRIVER_NAME "virtio-sst"
#define SST_DEVICE_NAME "sst"
#define SST_MAX_DEVICES 16
#define SST_MINOR_BASE  0

/*
 * Request State
 */
enum sst_request_state {
	SST_REQ_STATE_FREE = 0,
	SST_REQ_STATE_PENDING,
	SST_REQ_STATE_COMPLETED,
	SST_REQ_STATE_ERROR,
};

/*
 * Per-Request Context
 */
struct sst_request_ctx {
	struct SSTRequest      req;
	struct SSTResponse     resp;
	enum sst_request_state state;
	wait_queue_head_t      wait;
	struct list_head       list;
};

/*
 * VirtIO SST Device Instance
 */
struct sst_virt_device {
	struct virtio_device* vdev;

	/* VirtQueues */
	struct virtqueue* req_vq;
	struct virtqueue* resp_vq;
	struct virtqueue* event_vq;

	/* Character Device */
	struct cdev    cdev;
	dev_t          devt;
	struct device* dev;
	int            minor;

	/* Device Configuration */
	struct SSTConfig config;
	u64              features;
	u32              device_id;

	/* Request Management */
	spinlock_t       req_lock;
	struct list_head pending_requests;
	struct list_head completed_requests;
	u64              next_request_id;

	/* Statistics */
	atomic64_t total_requests;
	atomic64_t total_responses;
	atomic64_t total_events;
	atomic64_t total_errors;

	/* Status */
	bool ready;
	bool suspended;
};

/*
 * File Operations Context
 */
struct sst_file_ctx {
	struct sst_virt_device* sdev;
	struct sst_request_ctx* active_req;
};

/*
 * Function Prototypes
 */

/* VirtIO Driver Callbacks */
int  sst_probe(struct virtio_device* vdev);
void sst_remove(struct virtio_device* vdev);
int  sst_validate(struct virtio_device* vdev);

/* VirtQueue Handlers */
void sst_request_done(struct virtqueue* vq);
void sst_response_done(struct virtqueue* vq);
void sst_event_done(struct virtqueue* vq);

/* Character Device Operations */
int      sst_open(struct inode* inode, struct file* filp);
int      sst_release(struct inode* inode, struct file* filp);
ssize_t  sst_read(struct file* filp, char __user* buf, size_t count, loff_t* pos);
ssize_t  sst_write(struct file* filp, const char __user* buf, size_t count, loff_t* pos);
long     sst_ioctl(struct file* filp, unsigned int cmd, unsigned long arg);
__poll_t sst_poll(struct file* filp, struct poll_table_struct* wait);

/* Request Management */
int sst_submit_request(struct sst_virt_device* sdev, struct SSTRequest* req, struct SSTResponse* resp);
struct sst_request_ctx* sst_alloc_request(struct sst_virt_device* sdev);
void                    sst_free_request(struct sst_request_ctx* req_ctx);

/* Device Management */
int  sst_init_vqs(struct sst_virt_device* sdev);
void sst_del_vqs(struct sst_virt_device* sdev);
int  sst_read_config(struct sst_virt_device* sdev);

/* Sysfs Attributes */
ssize_t sst_sysfs_show_stats(struct device* dev, struct device_attribute* attr, char* buf);
ssize_t sst_sysfs_show_config(struct device* dev, struct device_attribute* attr, char* buf);

#endif /* SST_DEVICE_H */
