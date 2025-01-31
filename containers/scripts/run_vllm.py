from datasets import load_dataset
from vllm import LLM, SamplingParams
import time
import json
import os

# Load and filter the dataset
dataset = load_dataset("MLCommons/peoples_speech", "validation")
dataset = dataset.filter(lambda example: example['duration_ms'] < 30000 and example['duration_ms'] > 10000)
data_subset = dataset["validation"]# .select(range(200))

# Model configuration
tensor_parallel_size = 1
model = "openai/whisper-large-v3"  # ["openai/whisper-small", "openai/whisper-large-v3-turbo"]
max_num_seqs = 100
gpu_memory_utilization = 0.95
enforce_eager = True

temperature = 0
top_p = 1.0
max_tokens =200

concurrency_levels = [1, 2, 4, 8, 16, 32]  # Adjust these values as needed

llm = LLM(
    model=model,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    enforce_eager=enforce_eager,
    max_num_seqs=max_num_seqs
)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
)

# Prepare the prompt batch
prompt_batch = [
    {
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": (sample["audio"]["array"], sample["audio"]["sampling_rate"])
        }
    }
    for sample in data_subset
]

total_audio_seconds = sum([sample["duration_ms"] for sample in data_subset]) / 1000
expected_output = [sample["text"] for sample in data_subset]

# Function to process a batch of requests
def process_batch(batch):
    return llm.generate(batch, sampling_params)

# Function to run the script with a given batch size
def run_with_batching(batch_size):
    start_time = time.time()

    # Split the prompt batch into smaller chunks
    batches = [prompt_batch[i:i + batch_size] for i in range(0, len(prompt_batch), batch_size)]
    outputs = []

    for batch in batches:
        outputs.extend(process_batch(batch))

    end_time = time.time()
    total_time = end_time - start_time

    # Collect performance metrics
    num_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
    throughput = num_tokens / total_time  # tokens per second
    latency = total_time / len(outputs)  # time per request

    performance_metrics = {
        "concurrency": batch_size,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "throughput": throughput,
        "latency": latency,
        "requests_processed": len(outputs),
        "seconds_transcribed_per_sec": total_audio_seconds / total_time
    }
    # Store the metrics in a JSON file
    os.makedirs('/tmp/output', exist_ok=True)
    with open(f"/tmp/output/output-{batch_size:03d}.json", "w") as f:
        json.dump(performance_metrics, f, indent=4)


for batch_size in concurrency_levels:
    print(f"Running with batch size: {batch_size}")
    run_with_batching(batch_size)
