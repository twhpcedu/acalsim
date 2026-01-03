/*
 * VirtIO SST Device for QEMU - Implementation
 *
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

#include "hw/virtio/virtio-sst.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "hw/qdev-properties.h"
#include "hw/virtio/virtio-access.h"
#include "qapi/error.h"
#include "qemu/iov.h"
#include "qemu/log.h"

/*
 * SST Socket Connection
 */
bool virtio_sst_connect(VirtIOSST* s, Error** errp) {
	struct sockaddr_un addr;
	int                ret;

	if (s->connected) { return true; }

	/* Create Unix socket */
	s->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (s->socket_fd < 0) {
		error_setg_errno(errp, errno, "Failed to create socket");
		return false;
	}

	/* Connect to SST */
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);

	ret = connect(s->socket_fd, (struct sockaddr*)&addr, sizeof(addr));
	if (ret < 0) {
		error_setg_errno(errp, errno, "Failed to connect to SST at %s", s->socket_path);
		close(s->socket_fd);
		s->socket_fd = -1;
		return false;
	}

	s->connected = true;
	qemu_log("VirtIO SST: Connected to SST at %s\n", s->socket_path);

	return true;
}

void virtio_sst_disconnect(VirtIOSST* s) {
	if (s->socket_fd >= 0) {
		close(s->socket_fd);
		s->socket_fd = -1;
	}
	s->connected = false;
}

/*
 * Send request to SST and receive response
 */
bool virtio_sst_send_request(VirtIOSST* s, struct SSTRequest* req, struct SSTResponse* resp) {
	ssize_t sent, received;

	if (!s->connected) {
		qemu_log("VirtIO SST: Not connected to SST\n");
		return false;
	}

	/* Send request */
	sent = write(s->socket_fd, req, sizeof(*req));
	if (sent != sizeof(*req)) {
		qemu_log("VirtIO SST: Failed to send request (sent %zd bytes)\n", sent);
		s->total_errors++;
		return false;
	}

	/* Receive response */
	received = read(s->socket_fd, resp, sizeof(*resp));
	if (received != sizeof(*resp)) {
		qemu_log("VirtIO SST: Failed to receive response (got %zd bytes)\n", received);
		s->total_errors++;
		return false;
	}

	return resp->status == SST_STATUS_OK;
}

/*
 * Process single request from virtqueue
 * Note: elem is popped from req_vq but response is pushed to resp_vq (split queue design)
 */
void virtio_sst_process_request(VirtIOSST* s, VirtQueue* vq, VirtQueueElement* elem) {
	VirtIODevice*       vdev = VIRTIO_DEVICE(s);
	struct SSTRequest*  req;
	struct SSTResponse* resp;
	size_t              req_size, resp_size;
	bool                success;

	/* Get request from virtqueue buffer */
	req_size = iov_size(elem->out_sg, elem->out_num);
	if (req_size < sizeof(struct SSTRequest)) {
		qemu_log("VirtIO SST: Request too small: %zu bytes\n", req_size);
		goto error;
	}

	req  = g_malloc(sizeof(struct SSTRequest));
	resp = g_malloc(sizeof(struct SSTResponse));

	/* Copy request from guest memory */
	iov_to_buf(elem->out_sg, elem->out_num, 0, req, sizeof(*req));

	qemu_log("VirtIO SST: Processing request type=%s id=%lu\n", sst_request_type_str(req->type), req->request_id);

	/* Initialize response */
	memset(resp, 0, sizeof(*resp));
	resp->request_id = req->request_id;
	resp->user_data  = req->user_data;

	/* Handle request based on type */
	switch (req->type) {
		case SST_REQ_NOOP:
			/* No operation - just return success */
			resp->status = SST_STATUS_OK;
			success      = true;
			break;

		case SST_REQ_ECHO:
			/* Echo request - copy data back */
			memcpy(resp->payload.data, req->payload.data, SST_MAX_DATA_SIZE);
			resp->status = SST_STATUS_OK;
			success      = true;
			break;

		case SST_REQ_GET_INFO:
			/* Return device information */
			resp->payload.info.version           = s->config.version;
			resp->payload.info.capabilities      = s->config.features;
			resp->payload.info.max_compute_units = 1000000;
			resp->payload.info.memory_size       = 0x10000000;  // 256MB
			resp->status                         = SST_STATUS_OK;
			success                              = true;
			break;

		case SST_REQ_COMPUTE:
		case SST_REQ_READ:
		case SST_REQ_WRITE:
		case SST_REQ_CONFIGURE:
			/* Forward to SST */
			success = virtio_sst_send_request(s, req, resp);
			if (!success) { resp->status = SST_STATUS_NO_DEVICE; }
			break;

		case SST_REQ_RESET:
			/* Reset device state */
			resp->status = SST_STATUS_OK;
			success      = true;
			break;

		default:
			qemu_log("VirtIO SST: Unknown request type: %u\n", req->type);
			resp->status = SST_STATUS_INVALID;
			success      = false;
			break;
	}

	/* Copy response to guest memory */
	resp_size = iov_size(elem->in_sg, elem->in_num);
	if (resp_size < sizeof(*resp)) {
		qemu_log("VirtIO SST: Response buffer too small: %zu < %zu\n", resp_size, sizeof(*resp));
		resp->status = SST_STATUS_ERROR;
	} else {
		iov_from_buf(elem->in_sg, elem->in_num, 0, resp, sizeof(*resp));
	}

	/* Push response and notify guest - MUST push to same queue we popped from */
	virtqueue_push(vq, elem, sizeof(*resp));
	virtio_notify(vdev, vq);

	s->total_responses++;

	g_free(req);
	g_free(resp);
	return;

error:
	/* Even on error, must push element back to queue */
	s->total_errors++;

	/* Allocate and initialize error response */
	resp = g_malloc(sizeof(struct SSTResponse));
	memset(resp, 0, sizeof(*resp));
	resp->status = SST_STATUS_ERROR;

	/* Copy error response to guest memory */
	resp_size = iov_size(elem->in_sg, elem->in_num);
	if (resp_size >= sizeof(*resp)) { iov_from_buf(elem->in_sg, elem->in_num, 0, resp, sizeof(*resp)); }

	/* Must push element back even on error */
	virtqueue_push(vq, elem, sizeof(*resp));
	virtio_notify(vdev, vq);

	g_free(resp);
}

/*
 * Request VirtQueue Handler
 */
void virtio_sst_handle_request(VirtIODevice* vdev, VirtQueue* vq) {
	VirtIOSST*        s = VIRTIO_SST(vdev);
	VirtQueueElement* elem;

	while ((elem = virtqueue_pop(vq, sizeof(VirtQueueElement)))) {
		virtio_sst_process_request(s, vq, elem);
		s->total_requests++;
	}
}

/*
 * Response VirtQueue Handler (not used - responses go through same queue)
 */
void virtio_sst_handle_response(VirtIODevice* vdev, VirtQueue* vq) {
	/* Response queue is write-only from device perspective */
	qemu_log("VirtIO SST: Warning - Response queue should not be kicked\n");
}

/*
 * Event VirtQueue Handler
 */
void virtio_sst_handle_event(VirtIODevice* vdev, VirtQueue* vq) {
	/* Event queue for async notifications - not implemented yet */
	qemu_log("VirtIO SST: Event queue handler called (not implemented)\n");
}

/*
 * Inject async event to guest
 */
void virtio_sst_inject_event(VirtIOSST* s, struct SSTEvent* event) {
	VirtIODevice*     vdev = VIRTIO_DEVICE(s);
	VirtQueueElement* elem;

	elem = virtqueue_pop(s->event_vq, sizeof(VirtQueueElement));
	if (!elem) {
		qemu_log("VirtIO SST: No buffers available for event\n");
		return;
	}

	/* Copy event to guest memory */
	iov_from_buf(elem->in_sg, elem->in_num, 0, event, sizeof(*event));

	/* Push event and notify guest */
	virtqueue_push(s->event_vq, elem, sizeof(*event));
	virtio_notify(vdev, s->event_vq);

	s->total_events++;
}

/*
 * Feature Negotiation
 */
uint64_t virtio_sst_get_features(VirtIODevice* vdev, uint64_t requested_features, Error** errp) {
	VirtIOSST* s        = VIRTIO_SST(vdev);
	uint64_t   features = 0;

	/* Offer SST-specific features */
	features |= SST_FEATURE_ECHO;
	features |= SST_FEATURE_COMPUTE;
	features |= SST_FEATURE_MEMORY;
	features |= SST_FEATURE_RESET;

	/* Event support optional based on connection */
	if (s->connected) { features |= SST_FEATURE_EVENTS; }

	/* Return intersection of requested and supported */
	return features & requested_features;
}

void virtio_sst_set_features(VirtIODevice* vdev, uint64_t features) {
	VirtIOSST* s = VIRTIO_SST(vdev);
	s->features  = features;
	qemu_log("VirtIO SST: Features set: 0x%lx\n", features);
}

/*
 * Configuration Space Access
 */
void virtio_sst_get_config(VirtIODevice* vdev, uint8_t* config_data) {
	VirtIOSST*        s      = VIRTIO_SST(vdev);
	struct SSTConfig* config = (struct SSTConfig*)config_data;

	/* Update config with current values */
	s->config.version    = SST_PROTOCOL_VERSION;
	s->config.device_id  = s->device_id;
	s->config.features   = s->features;
	s->config.max_queues = 1;  // Single queue pair for now

	memcpy(config, &s->config, sizeof(s->config));
}

void virtio_sst_set_config(VirtIODevice* vdev, const uint8_t* config_data) {
	/* Configuration is read-only from driver perspective */
	qemu_log("VirtIO SST: Warning - Configuration is read-only\n");
}

/*
 * Device Status
 */
void virtio_sst_set_status(VirtIODevice* vdev, uint8_t status) {
	if (status & VIRTIO_CONFIG_S_DRIVER_OK) { qemu_log("VirtIO SST: Driver initialized successfully\n"); }

	if (!(status & VIRTIO_CONFIG_S_DRIVER_OK) && (status & VIRTIO_CONFIG_S_DRIVER)) {
		qemu_log("VirtIO SST: Driver removed\n");
	}
}

/*
 * Device Reset
 */
void virtio_sst_reset(VirtIODevice* vdev) {
	VirtIOSST* s = VIRTIO_SST(vdev);

	qemu_log("VirtIO SST: Reset device\n");

	/* Clear statistics */
	s->total_requests  = 0;
	s->total_responses = 0;
	s->total_events    = 0;
	s->total_errors    = 0;

	/* Reset features */
	s->features = 0;
}

/*
 * Device Realization
 */
void virtio_sst_realize(DeviceState* dev, Error** errp) {
	VirtIODevice* vdev = VIRTIO_DEVICE(dev);
	VirtIOSST*    s    = VIRTIO_SST(dev);

	qemu_log("VirtIO SST: Initializing device (socket=%s, id=%u)\n", s->socket_path, s->device_id);

	/* Initialize VirtIO device */
	virtio_init(vdev, "virtio-sst", VIRTIO_ID_SST, sizeof(struct SSTConfig));

	/* Create VirtQueues */
	s->req_vq   = virtio_add_queue(vdev, VIRTIO_SST_QUEUE_SIZE, virtio_sst_handle_request);
	s->resp_vq  = virtio_add_queue(vdev, VIRTIO_SST_QUEUE_SIZE, virtio_sst_handle_response);
	s->event_vq = virtio_add_queue(vdev, VIRTIO_SST_QUEUE_SIZE, virtio_sst_handle_event);

	/* Initialize request tracking */
	QLIST_INIT(&s->pending_requests);
	s->next_request_id = 1;

	/* Initialize statistics */
	s->total_requests  = 0;
	s->total_responses = 0;
	s->total_events    = 0;
	s->total_errors    = 0;

	/* Try to connect to SST (optional - can connect later) */
	if (s->socket_path && strlen(s->socket_path) > 0) {
		Error* local_err = NULL;
		if (!virtio_sst_connect(s, &local_err)) {
			qemu_log("VirtIO SST: Warning - %s\n", error_get_pretty(local_err));
			qemu_log("VirtIO SST: Device will work without SST connection\n");
			error_free(local_err);
		}
	}

	qemu_log("VirtIO SST: Device initialized successfully\n");
}

/*
 * Device Unrealization
 */
void virtio_sst_unrealize(DeviceState* dev) {
	VirtIODevice* vdev = VIRTIO_DEVICE(dev);
	VirtIOSST*    s    = VIRTIO_SST(dev);

	qemu_log("VirtIO SST: Shutting down\n");
	qemu_log(
	    "VirtIO SST: Statistics - Requests: %lu, Responses: %lu, "
	    "Events: %lu, Errors: %lu\n",
	    s->total_requests, s->total_responses, s->total_events, s->total_errors);

	/* Disconnect from SST */
	virtio_sst_disconnect(s);

	/* Delete VirtQueues */
	virtio_del_queue(vdev, 0); /* req_vq */
	virtio_del_queue(vdev, 1); /* resp_vq */
	virtio_del_queue(vdev, 2); /* event_vq */

	/* Cleanup VirtIO device */
	virtio_cleanup(vdev);
}

/*
 * Device Properties
 */
static Property virtio_sst_properties[] = {
    DEFINE_PROP_STRING("socket", VirtIOSST, socket_path),
    DEFINE_PROP_UINT32("device-id", VirtIOSST, device_id, 0),
    DEFINE_PROP_END_OF_LIST(),
};

/*
 * Device Class Initialization
 */
static void virtio_sst_class_init(ObjectClass* klass, void* data) {
	DeviceClass*       dc  = DEVICE_CLASS(klass);
	VirtioDeviceClass* vdc = VIRTIO_DEVICE_CLASS(klass);

	dc->desc = "VirtIO SST Device";
	device_class_set_props(dc, virtio_sst_properties);
	set_bit(DEVICE_CATEGORY_MISC, dc->categories);

	vdc->realize      = virtio_sst_realize;
	vdc->unrealize    = virtio_sst_unrealize;
	vdc->reset        = virtio_sst_reset;
	vdc->get_features = virtio_sst_get_features;
	vdc->set_features = virtio_sst_set_features;
	vdc->get_config   = virtio_sst_get_config;
	vdc->set_config   = virtio_sst_set_config;
	vdc->set_status   = virtio_sst_set_status;
}

/*
 * Device Type Information
 */
static const TypeInfo virtio_sst_info = {
    .name          = TYPE_VIRTIO_SST,
    .parent        = TYPE_VIRTIO_DEVICE,
    .instance_size = sizeof(VirtIOSST),
    .class_init    = virtio_sst_class_init,
};

/*
 * Module Registration
 */
static void virtio_sst_register_types(void) { type_register_static(&virtio_sst_info); }

type_init(virtio_sst_register_types)
