# SPDX-License-Identifier: Apache-2.0
"""
Evaluate Transcription API correctness by computing Word Error Rate (WER)
on a given ASR dataset. When provided, it will also compare the WER against
a baseline.
"""

import aiohttp
import asyncio
import json
import time

from transformers import AutoTokenizer
from typing import List
import numpy as np

from helpers import (data_generator, run_evaluation, load_hf_dataset)

CONCURRENT_REQUESTS = 2
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "D4nt3/esb-datasets-earnings22-validation-tiny-filtered"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

async def iterate_response(response) -> str:
    results = []
    if response.status == 200:
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
            if chunk != "[DONE]":
                results.append(dict(
                    time = time.perf_counter(),
                    text = json.loads(chunk)["text"]
                ))
    
    return results

async def transcribe(base_url: str,
                     waveform: np.ndarray,
                     sampling_rate: int,
                     reference: str):

    start = time.perf_counter()
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        url = f"{base_url}/generate_from_waveform"
        data = {"waveform_bytes": waveform.tobytes(), "sampling_rate": str(sampling_rate)}
        async with session.post(url, data=data) as response:
            results = await iterate_response(response)

    end = time.perf_counter()

    transcribed_text = "".join([res["text"] for res in results])
    print(f"transcribed_text: {transcribed_text}")
    req_results = dict(
        start = start,
        end = end,
        text = transcribed_text,
        reference = reference,
        decode_times = [res["time"] for res in results],
    )
    
    return req_results


async def bound_transcribe(
    sem: asyncio.Semaphore,
    base_url: str,
    waveform: np.ndarray,
    sampling_rate: int,
    reference: str
):
    # Use semaphore to limit concurrent requests.
    async with sem:
        return await transcribe(base_url, waveform, sampling_rate, reference)

async def process_dataset(
    dataset,
    concurrent_requests,
    base_url="http://localhost:8000"
):
    tasks: List[asyncio.Task] = []
    sem = asyncio.Semaphore(concurrent_requests)
    for waveform, sampling_rate, reference in data_generator(dataset):
        task = asyncio.create_task(
            bound_transcribe(sem, base_url, waveform, sampling_rate, reference))
        tasks.append(task)
    return await asyncio.gather(*tasks)


if __name__ == "__main__":    
    dataset = load_hf_dataset(DATASET_NAME)
    wer = run_evaluation(dataset, CONCURRENT_REQUESTS, process_dataset)
