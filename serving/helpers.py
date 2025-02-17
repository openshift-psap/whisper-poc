import asyncio
import evaluate
import time

import numpy as np

from datasets import load_dataset
from statistics import mean, median
from typing import Callable

WHISPER_SAMPLING_RATE = 16000

def data_generator(dataset):
    for sample in dataset:
        waveform = sample["audio"]["array"].astype(np.float32)
        sampling_rate = sample["audio"]["sampling_rate"]
        reference = sample["text"]
        yield (waveform, sampling_rate, reference)


def print_performance_metrics(results, total_time):
    latencies = [res[0] for res in results]
    total_tokens = sum([res[1] for res in results])

    total = len(results)
    print(f"Total Requests: {total}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Average Latency: {mean(latencies):.4f} seconds")
    print(f"Median Latency: {median(latencies):.4f} seconds")
    perc = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    print(f"95th Percentile Latency: {perc:.4f} seconds")
    # Throughput
    req_throughput = len(latencies) / total_time
    print(f"Estimated req_Throughput: {req_throughput:.2f} requests/s")
    throughput = total_tokens / total_time
    print(f"Estimated Throughput: {throughput:.2f} tok/s")


def load_hf_dataset(dataset_repo: str,
                    split="validation",
                    **hf_kwargs):
    ## Load and filter the dataset
    dataset = load_dataset(dataset_repo,
                           split=split,
                           **hf_kwargs)
    # Whisper max supported duration
    dataset = dataset.filter(lambda example: example['duration_ms'] < 30000)

    return dataset


def run_evaluation(dataset,
                   process_dataset_fn: Callable,
                   n_examples: int = -1,
                   print_metrics: bool = True):
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))

    # Warmup
    _ = asyncio.run(
        process_dataset_fn(dataset.select(range(1))))

    # Compute Latency.
    start = time.perf_counter()
    results = asyncio.run(process_dataset_fn(dataset))
    end = time.perf_counter()
    total_time = end - start
    print(f"Total Test Time: {total_time:.4f} seconds")
    if print_metrics:
        print_performance_metrics(results, total_time)
    
    # Compute WER
    predictions = [res[2] for res in results]
    references = [res[3] for res in results]
    wer = evaluate.load("wer")
    wer_score = 100 * wer.compute(references=references,
                                  predictions=predictions)
    print("WER:", wer_score)
    return wer_score