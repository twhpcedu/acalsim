/*
 * VirtIO SST Device Driver - Implementation
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

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/poll.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "sst-device.h"
MODULE_AUTHOR("Playlab/ACAL");
MODULE_DESCRIPTION("VirtIO SST Device Driver");
MODULE_VERSION("1.0");

/*
 * Global Driver State
 */
static struct class* sst_class;
static dev_t         sst_devt_base;
static DEFINE_IDA(sst_minor_ida);

/*
 * VirtQueue Callback - Response Received
 */
void sst_response_done(struct virtqueue* vq) {
	struct sst_virt_device* sdev = vq->vdev->priv;
	struct sst_request_ctx* req_ctx;
	unsigned int            len;
	unsigned long           flags;

	pr_debug("%s: Response received\n", SST_DRIVER_NAME);

	spin_lock_irqsave(&sdev->req_lock, flags);

	while ((req_ctx = virtqueue_get_buf(vq, &len)) != NULL) {
		/* Mark request as completed */
		req_ctx->state = SST_REQ_STATE_COMPLETED;

		/* Move to completed list */
		list_del(&req_ctx->list);
		list_add_tail(&req_ctx->list, &sdev->completed_requests);

		/* Wake up waiting process */
		wake_up(&req_ctx->wait);

		atomic64_inc(&sdev->total_responses);
	}

	spin_unlock_irqrestore(&sdev->req_lock, flags);
}

/*
 * VirtQueue Callback - Event Received
 */
void sst_event_done(struct virtqueue* vq) {
	struct sst_virt_device* sdev = vq->vdev->priv;

	pr_debug("%s: Event received (not implemented)\n", SST_DRIVER_NAME);
	atomic64_inc(&sdev->total_events);
}

/*
 * Submit Request to VirtQueue
 */
int sst_submit_request(struct sst_virt_device* sdev, struct SSTRequest* req, struct SSTResponse* resp) {
	struct sst_request_ctx* req_ctx;
	struct scatterlist      sg_out, sg_in;
	struct scatterlist*     sgs[2];
	unsigned long           flags;
	int                     ret;

	/* Allocate request context */
	req_ctx = kzalloc(sizeof(*req_ctx), GFP_KERNEL);
	if (!req_ctx) return -ENOMEM;

	/* Initialize request context */
	memcpy(&req_ctx->req, req, sizeof(*req));
	req_ctx->state = SST_REQ_STATE_PENDING;
	init_waitqueue_head(&req_ctx->wait);
	INIT_LIST_HEAD(&req_ctx->list);

	/* Assign request ID */
	spin_lock_irqsave(&sdev->req_lock, flags);
	req_ctx->req.request_id = sdev->next_request_id++;
	list_add_tail(&req_ctx->list, &sdev->pending_requests);
	spin_unlock_irqrestore(&sdev->req_lock, flags);

	/* Setup scatter-gather lists */
	sg_init_one(&sg_out, &req_ctx->req, sizeof(struct SSTRequest));
	sg_init_one(&sg_in, &req_ctx->resp, sizeof(struct SSTResponse));
	sgs[0] = &sg_out;
	sgs[1] = &sg_in;

	/* Add buffers to virtqueue */
	spin_lock_irqsave(&sdev->req_lock, flags);
	ret = virtqueue_add_sgs(sdev->req_vq, sgs, 1, 1, req_ctx, GFP_ATOMIC);
	if (ret < 0) {
		pr_err("%s: Failed to add request to virtqueue: %d\n", SST_DRIVER_NAME, ret);
		list_del(&req_ctx->list);
		spin_unlock_irqrestore(&sdev->req_lock, flags);
		kfree(req_ctx);
		return ret;
	}

	/* Kick virtqueue to notify device */
	virtqueue_kick(sdev->req_vq);
	spin_unlock_irqrestore(&sdev->req_lock, flags);

	atomic64_inc(&sdev->total_requests);

	/* Wait for response */
	ret = wait_event_interruptible_timeout(req_ctx->wait, req_ctx->state == SST_REQ_STATE_COMPLETED,
	                                       msecs_to_jiffies(5000));
	if (ret == 0) {
		pr_err("%s: Request timed out\n", SST_DRIVER_NAME);
		atomic64_inc(&sdev->total_errors);
		return -ETIMEDOUT;
	} else if (ret < 0) {
		pr_err("%s: Wait interrupted: %d\n", SST_DRIVER_NAME, ret);
		return ret;
	}

	/* Copy response back */
	memcpy(resp, &req_ctx->resp, sizeof(*resp));

	/* Cleanup */
	spin_lock_irqsave(&sdev->req_lock, flags);
	list_del(&req_ctx->list);
	spin_unlock_irqrestore(&sdev->req_lock, flags);
	kfree(req_ctx);

	pr_debug("%s: Request completed, status=%s\n", SST_DRIVER_NAME, sst_status_str(resp->status));

	return 0;
}

/*
 * Character Device - Open
 */
int sst_open(struct inode* inode, struct file* filp) {
	struct sst_virt_device* sdev;
	struct sst_file_ctx*    fctx;

	sdev = container_of(inode->i_cdev, struct sst_virt_device, cdev);

	if (!sdev->ready) {
		pr_err("%s: Device not ready\n", SST_DRIVER_NAME);
		return -ENODEV;
	}

	/* Allocate file context */
	fctx = kzalloc(sizeof(*fctx), GFP_KERNEL);
	if (!fctx) return -ENOMEM;

	fctx->sdev         = sdev;
	fctx->active_req   = NULL;
	filp->private_data = fctx;

	pr_debug("%s: Device opened\n", SST_DRIVER_NAME);
	return 0;
}

/*
 * Character Device - Release
 */
int sst_release(struct inode* inode, struct file* filp) {
	struct sst_file_ctx* fctx = filp->private_data;

	if (fctx) {
		kfree(fctx);
		filp->private_data = NULL;
	}

	pr_debug("%s: Device closed\n", SST_DRIVER_NAME);
	return 0;
}

/*
 * Character Device - Write (Send Request)
 */
ssize_t sst_write(struct file* filp, const char __user* buf, size_t count, loff_t* pos) {
	struct sst_file_ctx*    fctx = filp->private_data;
	struct sst_virt_device* sdev = fctx->sdev;
	struct SSTRequest       req;
	struct SSTResponse      resp;
	int                     ret;

	if (count < sizeof(struct SSTRequest)) {
		pr_err("%s: Write buffer too small\n", SST_DRIVER_NAME);
		return -EINVAL;
	}

	/* Copy request from userspace */
	if (copy_from_user(&req, buf, sizeof(req))) {
		pr_err("%s: Failed to copy request from user\n", SST_DRIVER_NAME);
		return -EFAULT;
	}

	pr_debug("%s: Write request type=%s\n", SST_DRIVER_NAME, sst_request_type_str(req.type));

	/* Submit request */
	ret = sst_submit_request(sdev, &req, &resp);
	if (ret < 0) return ret;

	/* Store response for subsequent read */
	fctx->active_req = kzalloc(sizeof(struct sst_request_ctx), GFP_KERNEL);
	if (!fctx->active_req) return -ENOMEM;

	memcpy(&fctx->active_req->resp, &resp, sizeof(resp));
	fctx->active_req->state = SST_REQ_STATE_COMPLETED;

	return sizeof(req);
}

/*
 * Character Device - Read (Get Response)
 */
ssize_t sst_read(struct file* filp, char __user* buf, size_t count, loff_t* pos) {
	struct sst_file_ctx* fctx = filp->private_data;
	size_t               to_copy;

	if (!fctx->active_req) {
		pr_debug("%s: No response available\n", SST_DRIVER_NAME);
		return -EAGAIN;
	}

	if (count < sizeof(struct SSTResponse)) {
		pr_err("%s: Read buffer too small\n", SST_DRIVER_NAME);
		return -EINVAL;
	}

	to_copy = min(count, sizeof(struct SSTResponse));

	/* Copy response to userspace */
	if (copy_to_user(buf, &fctx->active_req->resp, to_copy)) {
		pr_err("%s: Failed to copy response to user\n", SST_DRIVER_NAME);
		return -EFAULT;
	}

	/* Clear active request */
	kfree(fctx->active_req);
	fctx->active_req = NULL;

	pr_debug("%s: Read response (%zu bytes)\n", SST_DRIVER_NAME, to_copy);
	return to_copy;
}

/*
 * Character Device - IOCTL
 */
long sst_ioctl(struct file* filp, unsigned int cmd, unsigned long arg) {
	/* TODO: Implement ioctl commands for device control */
	pr_debug("%s: IOCTL cmd=%u (not implemented)\n", SST_DRIVER_NAME, cmd);
	return -ENOTTY;
}

/*
 * Character Device - Poll
 */
__poll_t sst_poll(struct file* filp, struct poll_table_struct* wait) {
	struct sst_file_ctx* fctx = filp->private_data;
	__poll_t             mask = 0;

	/* Always writable (can submit requests) */
	mask |= EPOLLOUT | EPOLLWRNORM;

	/* Readable if response available */
	if (fctx->active_req && fctx->active_req->state == SST_REQ_STATE_COMPLETED) mask |= EPOLLIN | EPOLLRDNORM;

	return mask;
}

/*
 * Character Device File Operations
 */
static const struct file_operations sst_fops = {
    .owner          = THIS_MODULE,
    .open           = sst_open,
    .release        = sst_release,
    .read           = sst_read,
    .write          = sst_write,
    .unlocked_ioctl = sst_ioctl,
    .poll           = sst_poll,
    .llseek         = no_llseek,
};

/*
 * Initialize VirtQueues
 */
int sst_init_vqs(struct sst_virt_device* sdev) {
	struct virtqueue* vqs[SST_VQ_MAX];
	vq_callback_t*    callbacks[SST_VQ_MAX];
	const char*       names[SST_VQ_MAX];
	int               ret;

	/* Setup queue callbacks */
	callbacks[SST_VQ_REQUEST]  = sst_response_done; /* Callback when request completes */
	callbacks[SST_VQ_RESPONSE] = NULL;              /* Response queue not used */
	callbacks[SST_VQ_EVENT]    = sst_event_done;

	names[SST_VQ_REQUEST]  = "request";
	names[SST_VQ_RESPONSE] = "response";
	names[SST_VQ_EVENT]    = "event";

	/* Find and initialize virtqueues */
	ret = virtio_find_vqs(sdev->vdev, SST_VQ_MAX, vqs, callbacks, names, NULL);
	if (ret) {
		pr_err("%s: Failed to find virtqueues: %d\n", SST_DRIVER_NAME, ret);
		return ret;
	}

	sdev->req_vq   = vqs[SST_VQ_REQUEST];
	sdev->resp_vq  = vqs[SST_VQ_RESPONSE];
	sdev->event_vq = vqs[SST_VQ_EVENT];

	pr_info("%s: VirtQueues initialized\n", SST_DRIVER_NAME);
	return 0;
}

/*
 * Delete VirtQueues
 */
void sst_del_vqs(struct sst_virt_device* sdev) { sdev->vdev->config->del_vqs(sdev->vdev); }

/*
 * Read Device Configuration
 */
int sst_read_config(struct sst_virt_device* sdev) {
	virtio_cread_bytes(sdev->vdev, 0, &sdev->config, sizeof(sdev->config));

	pr_info("%s: Device config - version=0x%x, features=0x%llx\n", SST_DRIVER_NAME, sdev->config.version,
	        sdev->config.features);

	return 0;
}

/*
 * VirtIO Probe - Device Discovery
 */
int sst_probe(struct virtio_device* vdev) {
	struct sst_virt_device* sdev;
	int                     ret, minor;

	pr_info("%s: Probing device\n", SST_DRIVER_NAME);

	/* Allocate device structure */
	sdev = kzalloc(sizeof(*sdev), GFP_KERNEL);
	if (!sdev) return -ENOMEM;

	sdev->vdev = vdev;
	vdev->priv = sdev;

	/* Initialize request management */
	spin_lock_init(&sdev->req_lock);
	INIT_LIST_HEAD(&sdev->pending_requests);
	INIT_LIST_HEAD(&sdev->completed_requests);
	sdev->next_request_id = 1;

	/* Initialize statistics */
	atomic64_set(&sdev->total_requests, 0);
	atomic64_set(&sdev->total_responses, 0);
	atomic64_set(&sdev->total_events, 0);
	atomic64_set(&sdev->total_errors, 0);

	/* Initialize virtqueues */
	ret = sst_init_vqs(sdev);
	if (ret) goto fail_vqs;

	/* Read device configuration */
	sst_read_config(sdev);

	/* Allocate character device minor number */
	minor = ida_simple_get(&sst_minor_ida, SST_MINOR_BASE, SST_MINOR_BASE + SST_MAX_DEVICES, GFP_KERNEL);
	if (minor < 0) {
		ret = minor;
		pr_err("%s: Failed to allocate minor number: %d\n", SST_DRIVER_NAME, ret);
		goto fail_minor;
	}
	sdev->minor = minor;
	sdev->devt  = MKDEV(MAJOR(sst_devt_base), minor);

	/* Create character device */
	cdev_init(&sdev->cdev, &sst_fops);
	sdev->cdev.owner = THIS_MODULE;

	ret = cdev_add(&sdev->cdev, sdev->devt, 1);
	if (ret) {
		pr_err("%s: Failed to add cdev: %d\n", SST_DRIVER_NAME, ret);
		goto fail_cdev;
	}

	/* Create device node */
	sdev->dev = device_create(sst_class, &vdev->dev, sdev->devt, sdev, "%s%d", SST_DEVICE_NAME, minor);
	if (IS_ERR(sdev->dev)) {
		ret = PTR_ERR(sdev->dev);
		pr_err("%s: Failed to create device: %d\n", SST_DRIVER_NAME, ret);
		goto fail_device;
	}

	/* Mark device as ready */
	sdev->ready = true;
	virtio_device_ready(vdev);

	pr_info("%s: Device /dev/%s%d registered successfully\n", SST_DRIVER_NAME, SST_DEVICE_NAME, minor);

	return 0;

fail_device:
	cdev_del(&sdev->cdev);
fail_cdev:
	ida_simple_remove(&sst_minor_ida, minor);
fail_minor:
	sst_del_vqs(sdev);
fail_vqs:
	kfree(sdev);
	return ret;
}

/*
 * VirtIO Remove - Device Removal
 */
void sst_remove(struct virtio_device* vdev) {
	struct sst_virt_device* sdev = vdev->priv;

	pr_info("%s: Removing device\n", SST_DRIVER_NAME);

	/* Mark device as not ready */
	sdev->ready = false;

	/* Remove device node */
	device_destroy(sst_class, sdev->devt);

	/* Delete character device */
	cdev_del(&sdev->cdev);

	/* Free minor number */
	ida_simple_remove(&sst_minor_ida, sdev->minor);

	/* Delete virtqueues */
	vdev->config->reset(vdev);
	sst_del_vqs(sdev);

	/* Print statistics */
	pr_info("%s: Statistics - Requests: %lld, Responses: %lld, Events: %lld, Errors: %lld\n", SST_DRIVER_NAME,
	        atomic64_read(&sdev->total_requests), atomic64_read(&sdev->total_responses),
	        atomic64_read(&sdev->total_events), atomic64_read(&sdev->total_errors));

	/* Free device structure */
	kfree(sdev);
}

/*
 * VirtIO Validate - Feature Negotiation
 */
int sst_validate(struct virtio_device* vdev) {
	/* No mandatory features for now */
	return 0;
}

/*
 * VirtIO Device ID Table
 */
static struct virtio_device_id id_table[] = {
    {VIRTIO_ID_SST, VIRTIO_DEV_ANY_ID},
    {0},
};
MODULE_DEVICE_TABLE(virtio, id_table);

/*
 * VirtIO Driver Structure
 */
static struct virtio_driver virtio_sst_driver = {
    .driver.name  = KBUILD_MODNAME,
    .driver.owner = THIS_MODULE,
    .id_table     = id_table,
    .probe        = sst_probe,
    .remove       = sst_remove,
    .validate     = sst_validate,
};

/*
 * Module Initialization
 */
static int __init sst_init(void) {
	int ret;

	pr_info("%s: Loading driver\n", SST_DRIVER_NAME);

	/* Allocate character device region */
	ret = alloc_chrdev_region(&sst_devt_base, SST_MINOR_BASE, SST_MAX_DEVICES, SST_DRIVER_NAME);
	if (ret) {
		pr_err("%s: Failed to allocate chrdev region: %d\n", SST_DRIVER_NAME, ret);
		return ret;
	}

	/* Create device class */
	sst_class = class_create(THIS_MODULE, SST_DRIVER_NAME);
	if (IS_ERR(sst_class)) {
		ret = PTR_ERR(sst_class);
		pr_err("%s: Failed to create class: %d\n", SST_DRIVER_NAME, ret);
		goto fail_class;
	}

	/* Register VirtIO driver */
	ret = register_virtio_driver(&virtio_sst_driver);
	if (ret) {
		pr_err("%s: Failed to register virtio driver: %d\n", SST_DRIVER_NAME, ret);
		goto fail_virtio;
	}

	pr_info("%s: Driver loaded successfully\n", SST_DRIVER_NAME);
	return 0;

fail_virtio:
	class_destroy(sst_class);
fail_class:
	unregister_chrdev_region(sst_devt_base, SST_MAX_DEVICES);
	return ret;
}

/*
 * Module Cleanup
 */
static void __exit sst_exit(void) {
	pr_info("%s: Unloading driver\n", SST_DRIVER_NAME);

	unregister_virtio_driver(&virtio_sst_driver);
	class_destroy(sst_class);
	unregister_chrdev_region(sst_devt_base, SST_MAX_DEVICES);

	pr_info("%s: Driver unloaded\n", SST_DRIVER_NAME);
}

module_init(sst_init);
module_exit(sst_exit);

MODULE_LICENSE("GPL");
