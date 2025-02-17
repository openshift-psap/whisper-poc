import asyncio
import evaluate
import time

import numpy as np
from transformers import AutoTokenizer

from datasets import load_dataset
from statistics import mean, median
from typing import Callable

WHISPER_SAMPLING_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


def data_generator(dataset):
    for sample in dataset:
        waveform = sample["audio"]["array"].astype(np.float32)
        sampling_rate = sample["audio"]["sampling_rate"]
        reference = sample["text"]
        yield (waveform, sampling_rate, reference)

def process_results(results):
    save_req_results = []
    for req_results in results:
        transcribed_text = req_results["text"]
        reference = req_results["reference"]
        num_tokens = len(
            TOKENIZER(transcribed_text, add_special_tokens=False).input_ids)
        out = TOKENIZER.normalize(transcribed_text)
        ref = TOKENIZER.normalize(reference)

        # Calculate per-token latencies
        prev_time = req_results["start"]
        decode_times = []
        for tok_time in req_results["decode_times"]:
            decode_times.append((tok_time - prev_time) * 1000)
            prev_time = tok_time

        print(f"results[-1]: {results[-1]}, num_tokens: {num_tokens}")
        print(f"ref: {ref}, out: {out}")
        
        ttft = decode_times[0]
        mean_itl = mean(decode_times)
        
        start_time = req_results["start"]
        end_time = req_results["end"]

        save_req_results.append(dict(
            num_tokens=num_tokens,
            out=out,
            ref=ref,
            start_time=start_time,
            end_time=end_time,
            latency=end_time - start_time,
            ttft=ttft,
            mean_itl=mean_itl,
            decode_times=decode_times,
        ))

    return(save_req_results)



def print_performance_metrics(results, total_time):
    ttfts = [res["ttft"] for res in results]
    decode_times = [tok_time for res in results for tok_time in res["decode_times"] ]
    itl_p95 = sorted(decode_times)[int(len(decode_times) * 0.95) - 1]
    print(f"95th Percentile ITL: {itl_p95:.4f} seconds")
    total_tokens = sum([res["num_tokens"] for res in results])
    latencies = [res["latency"] for res in results]

    total = len(results)
    print(f"Total Requests: {total}")
    print(f"Successful Requests: {len(latencies)}")
    print(f"Mean TTFT: {mean(ttfts)}")
    print(f"Mean ITL: {mean(decode_times)}")
    print(f"Average Latency: {mean(latencies):.4f} seconds")

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
                   concurrent_requests: int,
                   process_dataset_fn: Callable,
                   n_examples: int = -1,
                   print_metrics: bool = True):
    if n_examples > 0:
        dataset = dataset.select(range(n_examples))

    # Warmup
    _ = asyncio.run(
        process_dataset_fn(dataset.select(range(1)), 1))

    # Compute Latency.
    start = time.perf_counter()
    results = asyncio.run(process_dataset_fn(dataset, concurrent_requests))

    end = time.perf_counter()

    total_time = end - start

    print(f"Total Test Time: {total_time:.4f} seconds")

    results = process_results(results)
    
    if print_metrics:
        print_performance_metrics(results, total_time)
    
    # Compute WER
    predictions = [res["out"] for res in results]
    references = [res["ref"] for res in results]
    wer = evaluate.load("wer")
    wer_score = 100 * wer.compute(references=references,
                                  predictions=predictions)
    print("WER:", wer_score)
    return wer_score
