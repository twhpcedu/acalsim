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

#ifndef __ACALSIM_MMIO_DEVICE_HH__
#define __ACALSIM_MMIO_DEVICE_HH__

#include "ACALSimDeviceComponent.hh"
#include <queue>
#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

namespace ACALSim {
namespace QEMUIntegration {

/**
 * @brief Interrupt event sent from Device to QEMU
 *
 * This event represents an interrupt request from a device to the CPU.
 * The interrupt can be level-triggered or edge-triggered depending on
 * the device implementation.
 */
class InterruptEvent : public SST::Event {
public:
	/**
	 * @brief Interrupt types
	 */
	enum class Type : uint8_t {
		ASSERT   = 0,  ///< Assert interrupt (raise IRQ line)
		DEASSERT = 1   ///< Deassert interrupt (lower IRQ line)
	};

	/**
	 * @brief Default constructor for serialization
	 */
	InterruptEvent() : irq_num_(0), type_(Type::ASSERT) {}

	/**
	 * @brief Constructor for interrupt event
	 * @param irq_num IRQ number (device-specific interrupt line)
	 * @param type Interrupt type (assert or deassert)
	 */
	InterruptEvent(uint32_t irq_num, Type type) : irq_num_(irq_num), type_(type) {}

	// Getters
	uint32_t getIrqNum() const { return irq_num_; }
	Type     getType() const { return type_; }
	bool     isAssert() const { return type_ == Type::ASSERT; }

	// SST Event interface
	void serialize_order(SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		ser& irq_num_;
		ser& type_;
	}

	ImplementSerializable(ACALSim::QEMUIntegration::InterruptEvent);

private:
	uint32_t irq_num_;  ///< IRQ number
	Type     type_;     ///< Interrupt type
};

/**
 * @brief ACALSim-based MMIO Device with Interrupt Support
 *
 * This component demonstrates a cycle-accurate MMIO device that:
 * - Handles load/store operations from QEMU via memory-mapped registers
 * - Generates interrupts to notify QEMU of events
 * - Models realistic device behavior with configurable latencies
 * - Implements a simple DMA-like operation with completion interrupt
 *
 * Device Register Map (4KB at base address):
 * ----------------------------------------------
 * Offset  | Name           | Access | Description
 * ----------------------------------------------
 * 0x00    | CTRL           | RW     | Control register
 *         |                |        |   [0]: Start operation
 *         |                |        |   [1]: Reset device
 *         |                |        |   [2]: Enable interrupts
 * 0x04    | STATUS         | R      | Status register
 *         |                |        |   [0]: Busy (operation in progress)
 *         |                |        |   [1]: Done (operation completed)
 *         |                |        |   [2]: Error
 * 0x08    | INT_STATUS     | R/W1C  | Interrupt status (Write 1 to Clear)
 *         |                |        |   [0]: Operation complete IRQ
 *         |                |        |   [1]: Error IRQ
 * 0x0C    | INT_ENABLE     | RW     | Interrupt enable mask
 *         |                |        |   [0]: Enable completion IRQ
 *         |                |        |   [1]: Enable error IRQ
 * 0x10    | SRC_ADDR       | RW     | Source address for DMA
 * 0x14    | DST_ADDR       | RW     | Destination address for DMA
 * 0x18    | LENGTH         | RW     | Transfer length in bytes
 * 0x1C    | LATENCY        | RW     | Operation latency (cycles)
 * 0x20    | DATA_IN        | W      | Data input (for simple ops)
 * 0x24    | DATA_OUT       | R      | Data output (for simple ops)
 * 0x28    | CYCLE_COUNT    | R      | Cycle counter
 *
 * Operation Flow:
 * 1. CPU writes configuration registers (SRC_ADDR, DST_ADDR, LENGTH, etc.)
 * 2. CPU writes CTRL[0]=1 to start operation
 * 3. Device sets STATUS[0]=1 (busy) and begins processing
 * 4. After LATENCY cycles, device completes operation
 * 5. Device sets STATUS[1]=1 (done), INT_STATUS[0]=1
 * 6. If interrupts enabled, device asserts IRQ to QEMU
 * 7. CPU reads STATUS/INT_STATUS to check completion
 * 8. CPU writes INT_STATUS[0]=1 to clear interrupt
 * 9. Device deasserts IRQ
 */
class ACALSimMMIODevice : public SST::Component {
public:
	// Register offsets
	static constexpr uint64_t REG_CTRL        = 0x00;
	static constexpr uint64_t REG_STATUS      = 0x04;
	static constexpr uint64_t REG_INT_STATUS  = 0x08;
	static constexpr uint64_t REG_INT_ENABLE  = 0x0C;
	static constexpr uint64_t REG_SRC_ADDR    = 0x10;
	static constexpr uint64_t REG_DST_ADDR    = 0x14;
	static constexpr uint64_t REG_LENGTH      = 0x18;
	static constexpr uint64_t REG_LATENCY     = 0x1C;
	static constexpr uint64_t REG_DATA_IN     = 0x20;
	static constexpr uint64_t REG_DATA_OUT    = 0x24;
	static constexpr uint64_t REG_CYCLE_COUNT = 0x28;

	// Control register bits
	static constexpr uint32_t CTRL_START     = (1 << 0);
	static constexpr uint32_t CTRL_RESET     = (1 << 1);
	static constexpr uint32_t CTRL_INT_EN    = (1 << 2);

	// Status register bits
	static constexpr uint32_t STATUS_BUSY    = (1 << 0);
	static constexpr uint32_t STATUS_DONE    = (1 << 1);
	static constexpr uint32_t STATUS_ERROR   = (1 << 2);

	// Interrupt bits
	static constexpr uint32_t INT_COMPLETE   = (1 << 0);
	static constexpr uint32_t INT_ERROR      = (1 << 1);

	/**
	 * @brief SST ELI registration
	 */
	SST_ELI_REGISTER_COMPONENT(ACALSimMMIODevice, "acalsim", "MMIODevice", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "ACALSim MMIO device with interrupt support", COMPONENT_CATEGORY_UNCATEGORIZED)

	/**
	 * @brief Parameter documentation
	 */
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"},
	                        {"base_addr", "Device base address", "0x10001000"},
	                        {"size", "Device memory size", "4096"},
	                        {"verbose", "Verbosity level (0-3)", "1"},
	                        {"default_latency", "Default operation latency in cycles", "100"},
	                        {"irq_num", "IRQ number for this device", "1"})

	/**
	 * @brief Port documentation
	 */
	SST_ELI_DOCUMENT_PORTS({"cpu_port", "Port for CPU (QEMU) MMIO communication", {"ACALSim.MemoryTransaction"}},
	                       {"irq_port", "Port for interrupt signaling to QEMU", {"ACALSim.Interrupt"}})

	/**
	 * @brief Statistics documentation
	 */
	SST_ELI_DOCUMENT_STATISTICS({"mmio_reads", "Number of MMIO read operations", "count", 1},
	                            {"mmio_writes", "Number of MMIO write operations", "count", 1},
	                            {"operations_completed", "Number of operations completed", "count", 1},
	                            {"interrupts_generated", "Number of interrupts generated", "count", 1},
	                            {"avg_operation_latency", "Average operation latency", "cycles", 2})

	/**
	 * @brief Constructor
	 * @param id Component ID
	 * @param params Component parameters
	 */
	ACALSimMMIODevice(SST::ComponentId_t id, SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~ACALSimMMIODevice() override;

	// SST Component interface
	void setup() override;
	void finish() override;

	/**
	 * @brief Clock tick handler
	 * @param cycle Current cycle
	 * @return true to continue, false to stop
	 */
	bool clockTick(SST::Cycle_t cycle);

	/**
	 * @brief Handle incoming memory transaction from QEMU
	 * @param ev Memory transaction event
	 */
	void handleMemoryTransaction(SST::Event* ev);

private:
	/**
	 * @brief Device operation state
	 */
	struct Operation {
		bool     active;
		uint64_t start_cycle;
		uint64_t end_cycle;
		uint32_t src_addr;
		uint32_t dst_addr;
		uint32_t length;
	};

	/**
	 * @brief Read from device register
	 * @param offset Register offset
	 * @return Register value
	 */
	uint32_t readRegister(uint64_t offset);

	/**
	 * @brief Write to device register
	 * @param offset Register offset
	 * @param value Value to write
	 */
	void writeRegister(uint64_t offset, uint32_t value);

	/**
	 * @brief Start device operation
	 */
	void startOperation();

	/**
	 * @brief Complete current operation
	 */
	void completeOperation();

	/**
	 * @brief Reset device state
	 */
	void resetDevice();

	/**
	 * @brief Generate interrupt to QEMU
	 * @param irq_bits Interrupt bits to set
	 */
	void generateInterrupt(uint32_t irq_bits);

	/**
	 * @brief Clear interrupt
	 * @param irq_bits Interrupt bits to clear
	 */
	void clearInterrupt(uint32_t irq_bits);

	/**
	 * @brief Update interrupt line based on current state
	 */
	void updateInterruptLine();

	// SST infrastructure
	SST::Output output_;          ///< Output handler for logging
	SST::Link*  cpu_link_;        ///< Link to CPU (QEMU)
	SST::Link*  irq_link_;        ///< Link for interrupt signaling

	// Statistics
	SST::Statistic<uint64_t>* stat_mmio_reads_;
	SST::Statistic<uint64_t>* stat_mmio_writes_;
	SST::Statistic<uint64_t>* stat_ops_completed_;
	SST::Statistic<uint64_t>* stat_interrupts_;
	SST::Statistic<uint64_t>* stat_avg_latency_;

	// Device configuration
	uint64_t base_addr_;          ///< Base address
	uint64_t size_;               ///< Memory size
	uint32_t verbose_;            ///< Verbosity level
	uint32_t irq_num_;            ///< IRQ number

	// Device registers
	uint32_t reg_ctrl_;           ///< Control register
	uint32_t reg_status_;         ///< Status register
	uint32_t reg_int_status_;     ///< Interrupt status
	uint32_t reg_int_enable_;     ///< Interrupt enable
	uint32_t reg_src_addr_;       ///< Source address
	uint32_t reg_dst_addr_;       ///< Destination address
	uint32_t reg_length_;         ///< Transfer length
	uint32_t reg_latency_;        ///< Operation latency
	uint32_t reg_data_in_;        ///< Data input
	uint32_t reg_data_out_;       ///< Data output

	// Device state
	Operation    current_op_;     ///< Current operation
	uint64_t     cycle_count_;    ///< Cycle counter
	bool         irq_asserted_;   ///< IRQ line state
	uint64_t     total_ops_;      ///< Total operations
	uint64_t     total_latency_;  ///< Total latency for stats
};

}  // namespace QEMUIntegration
}  // namespace ACALSim

#endif  // __ACALSIM_MMIO_DEVICE_HH__
