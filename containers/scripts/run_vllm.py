from datasets import load_dataset
from vllm import LLM, SamplingParams
import time
import json
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process only the model name and range.")
parser.add_argument('--model', type=str, required=True, help='Specify the model')
parser.add_argument('--range', type=int, default=None, help='Specify the data subset range (optional)')
args = parser.parse_args()

# Load and filter the dataset
dataset = load_dataset("MLCommons/peoples_speech", "validation")
dataset = dataset.filter(lambda example: 10000 < example['duration_ms'] < 30000)

# Apply range selection if provided
data_subset = dataset["validation"].select(range(args.range)) if args.range else dataset["validation"]

temperature = 0
top_p = 1.0
max_tokens = 200
concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

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

total_audio_seconds = sum(sample["duration_ms"] for sample in data_subset) / 1000

def process_batch(batch, llm):
    return llm.generate(batch, sampling_params)

def run_with_batching(batch_size, llm):
    start_time = time.time()
    batches = [prompt_batch[i:i + batch_size] for i in range(0, len(prompt_batch), batch_size)]
    outputs = []

    for batch in batches:
        outputs.extend(process_batch(batch, llm))

    total_time = time.time() - start_time
    num_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
    throughput = num_tokens / total_time
    latency = total_time / len(outputs)

    performance_metrics = {
        "concurrency": batch_size,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "throughput": throughput,
        "latency": latency,
        "requests_processed": len(outputs),
        "seconds_transcribed_per_sec": total_audio_seconds / total_time,
        "start_time": start_time,
        "end_time": time.time()
    }

    os.makedirs('/tmp/output', exist_ok=True)
    with open(f"/tmp/output/output-vllm-{batch_size:03d}.json", "w") as f:
        json.dump(performance_metrics, f, indent=4)

def main():
    print(f"Model: {args.model}")
    model = f"openai/whisper-{args.model}"

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_num_seqs=100
    )

    for batch_size in concurrency_levels:
        print(f"Running with batch size: {batch_size}")
        run_with_batching(batch_size, llm)

if __name__ == "__main__":
    main()
