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

#include "QEMURealComponent.hh"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

using namespace ACALSim::QEMUReal;
using namespace ACALSim::QEMUIntegration;  // For event classes

// Constructor
QEMURealComponent::QEMURealComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      current_cycle_(0),
      qemu_pid_(-1),
      state_(QEMUState::IDLE),
      serial_fd_(-1),
      serial_ready_(false),
      next_req_id_(1),
      total_commands_(0),
      total_writes_(0),
      total_reads_(0),
      successful_transactions_(0),
      failed_transactions_(0) {
	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("QEMUReal[@p:@l]: ", verbose, 0, SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing QEMU Real Component\n");

	// Get parameters
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	binary_path_           = params.find<std::string>("binary_path", "");
	qemu_path_             = params.find<std::string>("qemu_path", "qemu-system-riscv32");
	socket_path_           = params.find<std::string>("socket_path", "/tmp/qemu-sst.sock");

	if (binary_path_.empty()) { out_.fatal(CALL_INFO, -1, "Error: binary_path parameter is required\n"); }

	out_.verbose(CALL_INFO, 1, 0, "Configuration:\n");
	out_.verbose(CALL_INFO, 1, 0, "  Clock: %s\n", clock_freq.c_str());
	out_.verbose(CALL_INFO, 1, 0, "  Binary: %s\n", binary_path_.c_str());
	out_.verbose(CALL_INFO, 1, 0, "  QEMU: %s\n", qemu_path_.c_str());
	out_.verbose(CALL_INFO, 1, 0, "  Socket: %s\n", socket_path_.c_str());

	// Register clock
	tc_ = registerClock(clock_freq, new SST::Clock::Handler<QEMURealComponent>(this, &QEMURealComponent::clockTick));

	// Configure link to device
	device_link_ = configureLink(
	    "device_port", new SST::Event::Handler<QEMURealComponent>(this, &QEMURealComponent::handleDeviceResponse));

	if (!device_link_) { out_.fatal(CALL_INFO, -1, "Error: Failed to configure device_port link\n"); }

	// Primary component
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	out_.verbose(CALL_INFO, 1, 0, "QEMUReal component initialized successfully\n");
}

// Destructor
QEMURealComponent::~QEMURealComponent() {
	terminateQEMU();

	if (serial_fd_ >= 0) { close(serial_fd_); }
}

// Setup
void QEMURealComponent::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");

	// Launch QEMU process
	launchQEMU();
}

// Finish
void QEMURealComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");
	out_.verbose(CALL_INFO, 1, 0, "Statistics:\n");
	out_.verbose(CALL_INFO, 1, 0, "  Total commands:     %lu\n", total_commands_);
	out_.verbose(CALL_INFO, 1, 0, "  Total writes:       %lu\n", total_writes_);
	out_.verbose(CALL_INFO, 1, 0, "  Total reads:        %lu\n", total_reads_);
	out_.verbose(CALL_INFO, 1, 0, "  Successful:         %lu\n", successful_transactions_);
	out_.verbose(CALL_INFO, 1, 0, "  Failed:             %lu\n", failed_transactions_);

	terminateQEMU();
}

// Clock tick
bool QEMURealComponent::clockTick(SST::Cycle_t cycle) {
	current_cycle_ = cycle;

	out_.verbose(CALL_INFO, 3, 0, "[CLOCK] Cycle %lu, State: %d\n", cycle, static_cast<int>(state_));

	// Monitor QEMU and handle serial data
	if (state_ == QEMUState::RUNNING || state_ == QEMUState::WAITING_DEVICE) {
		monitorQEMU();
		if (serial_ready_) { handleSerialData(); }
	}

	// Check if we're done
	bool is_done = (state_ == QEMUState::COMPLETED || state_ == QEMUState::ERROR);

	if (is_done) { primaryComponentOKToEndSim(); }

	return false;  // false = continue ticking
}

// Handle device response
void QEMURealComponent::handleDeviceResponse(SST::Event* ev) {
	MemoryResponseEvent* resp = dynamic_cast<MemoryResponseEvent*>(ev);
	if (!resp) {
		out_.fatal(CALL_INFO, -1, "Error: Received invalid event type\n");
		delete ev;
		return;
	}

	uint64_t req_id = resp->getReqId();

	out_.verbose(CALL_INFO, 2, 0, "Received device response: req_id=%lu data=0x%x success=%d\n", req_id,
	             resp->getData(), resp->getSuccess());

	// Find pending request
	auto it = pending_requests_.find(req_id);
	if (it == pending_requests_.end()) {
		out_.verbose(CALL_INFO, 2, 0, "Warning: Response for unknown request %lu\n", req_id);
		delete ev;
		return;
	}

	PendingQEMURequest& req = it->second;

	// Format response for QEMU
	std::string response;
	if (resp->getSuccess()) {
		char hex_data[16];
		snprintf(hex_data, sizeof(hex_data), "%08X", resp->getData());
		response = "SST:OK:" + std::string(hex_data) + "\n";
		successful_transactions_++;
	} else {
		response = "SST:ERR:0001\n";
		failed_transactions_++;
	}

	// Send response to QEMU
	out_.verbose(CALL_INFO, 2, 0, "Sending response to QEMU: %s", response.c_str());
	sendSerialResponse(response);

	// Remove from pending
	pending_requests_.erase(it);

	// Update state
	if (pending_requests_.empty()) { setState(QEMUState::RUNNING); }

	delete ev;
}

// Launch QEMU
void QEMURealComponent::launchQEMU() {
	out_.verbose(CALL_INFO, 1, 0, "Launching QEMU process...\n");

	setState(QEMUState::LAUNCHING);

	// Remove existing socket
	unlink(socket_path_.c_str());

	// Setup server socket FIRST (before forking)
	// Create server socket
	int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (server_fd < 0) { out_.fatal(CALL_INFO, -1, "Error: Failed to create server socket\n"); }

	// Bind to socket path
	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

	if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		close(server_fd);
		out_.fatal(CALL_INFO, -1, "Error: Failed to bind server socket: %s\n", strerror(errno));
	}

	// Listen for connections
	if (listen(server_fd, 1) < 0) {
		close(server_fd);
		out_.fatal(CALL_INFO, -1, "Error: Failed to listen on socket: %s\n", strerror(errno));
	}

	out_.verbose(CALL_INFO, 1, 0, "Server socket listening at %s\n", socket_path_.c_str());

	// Fork QEMU process
	qemu_pid_ = fork();

	if (qemu_pid_ < 0) {
		close(server_fd);
		out_.fatal(CALL_INFO, -1, "Error: Failed to fork QEMU process\n");
	}

	if (qemu_pid_ == 0) {
		// Child process - exec QEMU as client
		// Close server socket in child
		close(server_fd);

		// QEMU will connect to SST's server socket
		std::string serial_arg = "unix:" + socket_path_;

		const char* args[] = {
		    qemu_path_.c_str(), "-M", "virt", "-bios", "none", "-nographic", "-kernel", binary_path_.c_str(), "-serial",
		    serial_arg.c_str(), NULL};

		execvp(qemu_path_.c_str(), const_cast<char* const*>(args));

		// If exec fails
		fprintf(stderr, "Failed to exec QEMU: %s\n", strerror(errno));
		exit(1);
	}

	// Parent process - accept connection from QEMU
	out_.verbose(CALL_INFO, 1, 0, "QEMU PID: %d\n", qemu_pid_);
	out_.verbose(CALL_INFO, 1, 0, "Waiting for QEMU to connect...\n");

	// Set socket to non-blocking for accept with timeout
	int flags = fcntl(server_fd, F_GETFL, 0);
	fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

	// Wait for QEMU to connect (with timeout)
	for (int i = 0; i < 50; i++) {
		serial_fd_ = accept(server_fd, NULL, NULL);
		if (serial_fd_ >= 0) {
			// Connection accepted!
			close(server_fd);

			// Set client socket to non-blocking
			flags = fcntl(serial_fd_, F_GETFL, 0);
			fcntl(serial_fd_, F_SETFL, flags | O_NONBLOCK);

			serial_ready_ = true;
			out_.verbose(CALL_INFO, 1, 0, "QEMU connected to serial socket\n");
			setState(QEMUState::RUNNING);
			return;
		}

		// Check if error is EAGAIN/EWOULDBLOCK (no connection yet)
		if (errno != EAGAIN && errno != EWOULDBLOCK) {
			close(server_fd);
			out_.fatal(CALL_INFO, -1, "Error: accept() failed: %s\n", strerror(errno));
		}

		// Wait and retry
		out_.verbose(CALL_INFO, 3, 0, "Waiting for QEMU connection (attempt %d/50)...\n", i + 1);
		usleep(200000);  // 200ms
	}

	close(server_fd);
	out_.fatal(CALL_INFO, -1, "Error: QEMU failed to connect after 50 attempts (10 seconds)\n");
}

// Setup serial connection - now handled in launchQEMU()
void QEMURealComponent::setupSerial() {
	// Serial setup is now integrated into launchQEMU()
	// This function kept for compatibility
	out_.verbose(CALL_INFO, 2, 0, "setupSerial() called (socket already set up in launchQEMU)\n");
}

// Monitor QEMU
void QEMURealComponent::monitorQEMU() {
	if (qemu_pid_ < 0) return;

	// Check if QEMU is still running
	int   status;
	pid_t result = waitpid(qemu_pid_, &status, WNOHANG);

	if (result > 0) {
		// QEMU exited
		if (WIFEXITED(status)) {
			int exit_code = WEXITSTATUS(status);
			out_.verbose(CALL_INFO, 1, 0, "QEMU exited with code %d\n", exit_code);

			if (exit_code == 0) {
				setState(QEMUState::COMPLETED);
			} else {
				setState(QEMUState::ERROR);
			}
		} else if (WIFSIGNALED(status)) {
			out_.verbose(CALL_INFO, 1, 0, "QEMU killed by signal %d\n", WTERMSIG(status));
			setState(QEMUState::ERROR);
		}

		qemu_pid_ = -1;
	}
}

// Handle serial data
void QEMURealComponent::handleSerialData() {
	char    buffer[1024];
	ssize_t bytes_read = read(serial_fd_, buffer, sizeof(buffer) - 1);

	if (bytes_read > 0) {
		buffer[bytes_read] = '\0';
		serial_buffer_ += buffer;

		// Process complete lines
		size_t pos;
		while ((pos = serial_buffer_.find('\n')) != std::string::npos) {
			std::string line = serial_buffer_.substr(0, pos);
			serial_buffer_   = serial_buffer_.substr(pos + 1);

			// Check if it's an SST command
			if (line.substr(0, 4) == "SST:") {
				parseCommand(line);
			} else {
				// Regular QEMU output - just log it
				out_.verbose(CALL_INFO, 2, 0, "[QEMU] %s\n", line.c_str());
			}
		}
	}
}

// Parse SST command
void QEMURealComponent::parseCommand(const std::string& line) {
	total_commands_++;

	out_.verbose(CALL_INFO, 2, 0, "Parsing SST command: %s\n", line.c_str());

	std::string operation;
	uint64_t    addr;
	uint32_t    data;

	if (!parseSSTCommand(line, operation, addr, data)) {
		out_.verbose(CALL_INFO, 1, 0, "Warning: Failed to parse SST command: %s\n", line.c_str());
		sendSerialResponse("SST:ERR:0002\n");
		return;
	}

	// Dispatch based on operation
	if (operation == "WRITE") {
		total_writes_++;
		sendDeviceRequest(TransactionType::STORE, addr, data);
	} else if (operation == "READ") {
		total_reads_++;
		sendDeviceRequest(TransactionType::LOAD, addr, 0);
	} else {
		out_.verbose(CALL_INFO, 1, 0, "Warning: Unknown SST operation: %s\n", operation.c_str());
		sendSerialResponse("SST:ERR:0003\n");
	}
}

// Parse SST command format
bool QEMURealComponent::parseSSTCommand(const std::string& cmd, std::string& operation, uint64_t& addr,
                                        uint32_t& data) {
	// Format: SST:WRITE:ADDR:DATA or SST:READ:ADDR:DATA
	size_t pos1 = cmd.find(':', 0);
	size_t pos2 = cmd.find(':', pos1 + 1);
	size_t pos3 = cmd.find(':', pos2 + 1);

	if (pos1 == std::string::npos || pos2 == std::string::npos || pos3 == std::string::npos) { return false; }

	operation            = cmd.substr(pos1 + 1, pos2 - pos1 - 1);
	std::string addr_str = cmd.substr(pos2 + 1, pos3 - pos2 - 1);
	std::string data_str = cmd.substr(pos3 + 1);

	// Parse hex values
	addr = std::stoull(addr_str, nullptr, 16);
	data = std::stoul(data_str, nullptr, 16);

	return true;
}

// Send device request
void QEMURealComponent::sendDeviceRequest(TransactionType type, uint64_t addr, uint32_t data) {
	uint64_t req_id = next_req_id_++;

	out_.verbose(CALL_INFO, 2, 0, "Sending device request: type=%d addr=0x%lx data=0x%x req_id=%lu\n",
	             static_cast<int>(type), addr, data, req_id);

	// Create transaction event
	auto* trans = new MemoryTransactionEvent(type, addr, data, 4, req_id);
	device_link_->send(trans);

	// Track pending request
	PendingQEMURequest pending;
	pending.sst_req_id        = req_id;
	pending.address           = addr;
	pending.type              = type;
	pending_requests_[req_id] = pending;

	setState(QEMUState::WAITING_DEVICE);
}

// Send serial response
void QEMURealComponent::sendSerialResponse(const std::string& response) {
	out_.verbose(CALL_INFO, 3, 0, "Sending serial response: %s", response.c_str());

	if (serial_fd_ >= 0) { write(serial_fd_, response.c_str(), response.length()); }
}

// Terminate QEMU
void QEMURealComponent::terminateQEMU() {
	if (qemu_pid_ > 0) {
		out_.verbose(CALL_INFO, 2, 0, "Terminating QEMU process (PID %d)\n", qemu_pid_);
		kill(qemu_pid_, SIGTERM);
		waitpid(qemu_pid_, NULL, 0);
		qemu_pid_ = -1;
	}
}

// Set state
void QEMURealComponent::setState(QEMUState new_state) {
	if (state_ != new_state) {
		out_.verbose(CALL_INFO, 3, 0, "State transition: %d -> %d\n", static_cast<int>(state_),
		             static_cast<int>(new_state));
		state_ = new_state;
	}
}
