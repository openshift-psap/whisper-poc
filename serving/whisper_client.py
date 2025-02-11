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

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "D4nt3/esb-datasets-earnings22-validation-tiny-filtered"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

async def iterate_response(response) -> str:
    output_text = ""
    if response.status == 200:
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
            if chunk != "[DONE]":
                output_text += json.loads(chunk)["text"]
    return output_text

async def _transcribe_from_waveform(base_url: str, waveform: np.array,
                                    sr: int) -> str:
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        url = f"{base_url}/generate_from_waveform"
        data = {"waveform_bytes": waveform.tobytes(), "sampling_rate": str(sr)}
        async with session.post(url, data=data) as response:
            return await iterate_response(response)

async def transcribe(base_url: str,
                     waveform: np.ndarray,
                     sampling_rate: int,
                     reference: str):

    start = time.perf_counter()
    transcribed_text = await _transcribe_from_waveform(
        base_url=base_url,
        waveform=waveform,
        sr=sampling_rate,
    )
    latency = time.perf_counter() - start

    num_tokens = len(
        TOKENIZER(transcribed_text, add_special_tokens=False).input_ids)

    # Normalize *english* output/reference for evaluation.
    out = TOKENIZER.normalize(transcribed_text)
    ref = TOKENIZER.normalize(reference)
    return latency, num_tokens, out, ref


async def process_dataset(
    dataset,
    base_url="http://localhost:8000"
):
    tasks: List[asyncio.Task] = []
    for waveform, sampling_rate, reference in data_generator(dataset):
        task = asyncio.create_task(
            transcribe(base_url, waveform, sampling_rate, reference))
        tasks.append(task)
    return await asyncio.gather(*tasks)


if __name__ == "__main__":    
    dataset = load_hf_dataset(DATASET_NAME)
    wer = run_evaluation(dataset, process_dataset)
