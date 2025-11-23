# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
LLAMA 2 7B Inference with SST Accelerator Integration

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import time

# Check if PyTorch is available
try:
	import torch
	from transformers import AutoTokenizer, AutoModelForCausalLM
	PYTORCH_AVAILABLE = True
except ImportError:
	PYTORCH_AVAILABLE = False
	print("WARNING: PyTorch or Transformers not available")
	print("This is a demonstration script.")
	print(
	    "For full functionality, install PyTorch and Transformers as described in PYTORCH_LLAMA_SETUP.md"
	)

from llama_sst_backend import SSTAcceleratorBackend

# Model path (on virtual disk)
MODEL_PATH = "/mnt/models/llama-2-7b"


def load_model():
	"""Load LLAMA 2 model and tokenizer"""
	if not PYTORCH_AVAILABLE:
		print("ERROR: PyTorch not available. Cannot load model.")
		sys.exit(1)

	print(f"Loading model from {MODEL_PATH}...")
	print("This may take several minutes...")

	try:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
		model = AutoModelForCausalLM.from_pretrained(
		    MODEL_PATH,
		    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
		    low_cpu_mem_usage=True,
		    device_map="auto"  # Automatic device placement
		)

		print("Model loaded successfully")
		print(f"Model parameters: {model.num_parameters():,}")
		print(f"Model dtype: {model.dtype}")

		return tokenizer, model

	except FileNotFoundError:
		print(f"ERROR: Model not found at {MODEL_PATH}")
		print("Please ensure:")
		print("  1. Virtual disk is mounted: mount /dev/vda /mnt/models")
		print("  2. Model files exist: ls /mnt/models/llama-2-7b/")
		sys.exit(1)
	except Exception as e:
		print(f"ERROR loading model: {e}")
		sys.exit(1)


def generate_text(prompt, tokenizer, model, backend, max_new_tokens=100):
	"""Generate text from prompt"""
	inputs = tokenizer(prompt, return_tensors="pt")

	print(f"\nPrompt: {prompt}")
	print(f"Generating {max_new_tokens} tokens...")
	print("")

	start_time = time.time()

	# Generate with SST backend tracking
	backend.reset_stats()

	try:
		outputs = model.generate(
		    inputs.input_ids,
		    max_new_tokens=max_new_tokens,
		    do_sample=True,
		    top_k=50,
		    top_p=0.95,
		    temperature=0.7,
		    pad_token_id=tokenizer.eos_token_id
		)

		generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

	except Exception as e:
		print(f"ERROR during generation: {e}")
		return None

	end_time = time.time()
	generation_time = end_time - start_time

	print(f"\nGeneration completed in {generation_time:.2f} seconds")

	# Print SST statistics
	backend.print_stats()

	return generated_text


def demo_mode():
	"""Demo mode when PyTorch is not available"""
	print("\n" + "=" * 60)
	print("DEMO MODE - PyTorch Not Available")
	print("=" * 60)
	print("\nThis demonstrates the SST accelerator integration.")
	print("For actual LLAMA 2 inference, install PyTorch as described")
	print("in PYTORCH_LLAMA_SETUP.md")

	backend = SSTAcceleratorBackend()

	print("\nSimulating inference with SST accelerators...")
	print("")

	# Simulate some operations
	for i in range(10):
		backend.simulate_attention_op(batch_size=1, seq_len=128)
		backend.simulate_ffn_op(batch_size=1, hidden_size=4096)

	print("\n" + "=" * 60)
	print("DEMO RESULTS (Simulated):")
	print("=" * 60)
	print("\nPrompt: Explain quantum computing in simple terms.")
	print("\nGenerated Text (example):")
	print("Quantum computing leverages quantum mechanical phenomena")
	print("like superposition and entanglement to process information...")
	print("=" * 60)

	backend.print_stats()


def main():
	print("=" * 60)
	print("LLAMA 2 7B Inference with SST Integration")
	print("=" * 60)
	print("")

	# Check command line arguments
	if len(sys.argv) < 2:
		print("Usage: llama_inference.py <prompt>")
		print("")
		print("Examples:")
		print("  ./llama_inference.py \"Explain quantum computing\"")
		print("  ./llama_inference.py \"What is machine learning?\"")
		print("")
		sys.exit(1)

	prompt = " ".join(sys.argv[1 :])

	# Initialize SST backend
	backend = SSTAcceleratorBackend()

	# Check if SST device is available
	if not backend.check_device():
		print("WARNING: SST device (/dev/sst0) not available")
		print("Statistics will be estimates only")
		print("")

	# Check if PyTorch is available
	if not PYTORCH_AVAILABLE:
		demo_mode()
		return

	# Load model
	tokenizer, model = load_model()

	# Instrument model with SST backend
	print("\nInstrumenting model with SST accelerator backend...")
	model = backend.instrument_model(model)
	print("Model instrumented")

	# Generate text
	result = generate_text(prompt, tokenizer, model, backend)

	if result:
		print("\n" + "=" * 60)
		print("GENERATED TEXT:")
		print("=" * 60)
		print(result)
		print("=" * 60)

	# Cleanup
	backend.close()


if __name__ == "__main__":
	main()
