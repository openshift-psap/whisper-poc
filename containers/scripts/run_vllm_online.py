# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
import time
import json
import os
from typing import List, Tuple, AsyncGenerator
from statistics import mean, median
import asyncio
from openai import OpenAI
from statistics import mean, median
import io
from scipy.io.wavfile import write
import numpy as np
from transformers import AutoTokenizer

# NOTE Start server with: `vllm serve openai/whisper-large-v3 --max-num-seqs 100 --gpu-memory-utilization 0.95 --enforce-eager`

# NOTE take into account overhead of loading and sending the file to an actual server (offline we're just passing pre-loaded arrays), 
# as well as batching

# Configuration
TOTAL_REQUESTS = 32
MAX_REQ_CONCURRENCY = 100

openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key="", base_url=openai_api_base)
model_name = "openai/whisper-large-v3"  # ["openai/whisper-small", "openai/whisper-large-v3-turbo"]
tokenizer = AutoTokenizer.from_pretrained(model_name)


def convert_to_bytes(y, sr):
    buffer = io.BytesIO()
    y = (y * 32767).astype(np.int16)  # Convert float to PCM 16-bit
    write(buffer, sr, y)
    buffer.seek(0)
    return buffer

async def transcribe_audio(client, y, sr):
    status = 200
    try:
        # Send loaded audio directly instead of loading from disk, dont account for that time though
        with convert_to_bytes(y, sr) as f:
            start_time = time.perf_counter()
            transcription = client.audio.transcriptions.create(
                file=f,
                model=model_name,
                language="en",
                # max_tokens=200, TODO cant set here
                temperature=0.0,
            ) 
            # NOTE there's no streaming in transcriptions, can't measure ttft 
    except Exception as e:
        print(f"Error: {e}")
        status = 500
    end_time = time.perf_counter()
    latency = end_time - start_time
    assert status==200
    num_output_tokens = len(tokenizer(transcription.text, add_special_tokens=False).input_ids)
    return latency, num_output_tokens

async def bound_transcribe(sem, client, audio):
    # Use semaphore to limit concurrent requests.
    async with sem:
        # return await asyncio.to_thread(transcribe_audio, client, *audio)
        return await transcribe_audio(client, *audio)

async def run_load_test(data, concurrent_request):
    sem = asyncio.Semaphore(concurrent_request)
    tasks: List[asyncio.Task] = []
    # tasks = [bound_transcribe(sem, client, (sample["audio"]["array"], sample["audio"]["sampling_rate"])) for sample in data]
    for sample in data:
        task = asyncio.create_task(bound_transcribe(sem, client, (sample["audio"]["array"], sample["audio"]["sampling_rate"])))
        tasks.append(task)
    return await asyncio.gather(*tasks)

def analyze_results(results):
    latencies = [latency for latency, _ in results]
    total_tokens = sum([nt for _, nt in results])

    total = len(results)
    print(f"Total Requests: {total}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Average Latency: {mean(latencies):.4f} seconds")
    print(f"Median Latency: {median(latencies):.4f} seconds")
    print(f"95th Percentile Latency: {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.4f} seconds")
    # Throughput
    total_time = sum(latencies)
    throughput = len(latencies) / total_time 
    print(f"Estimated Throughput: {throughput:.2f} requests/s")
    throughput = total_tokens / total_time 
    print(f"Estimated Throughput: {throughput:.2f} tok/s")

## Load and filter the dataset
dataset = load_dataset("MLCommons/peoples_speech", "validation")
dataset = dataset.filter(lambda example: example['duration_ms'] < 30000 and example['duration_ms'] > 10000)
data_subset = dataset["validation"]# .select(range(200))
if TOTAL_REQUESTS > 0:
    data_subset = data_subset.select(range(TOTAL_REQUESTS))


start = time.perf_counter()
results = asyncio.run(run_load_test(data_subset, MAX_REQ_CONCURRENCY))
end = time.perf_counter()
total_time = end - start
print(f"Total Test Time: {total_time:.4f} seconds")
analyze_results(results)