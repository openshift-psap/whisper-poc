# SPDX-License-Identifier: Apache-2.0
"""
Benchmark OpenAI-compatible Whisper API for streaming speech-to-text.
Measures Word Error Rate (WER) and per-token/chunk latencies.
Handles datasets providing audio as NumPy arrays.
"""

import asyncio
import json
import time
import io
import wave 
import os

import httpx
import numpy as np
from typing import List, Dict, Any, AsyncGenerator, Generator

# Assuming helpers.py is in the same directory or accessible in PYTHONPATH
from helpers import (run_evaluation, load_hf_dataset, TOKENIZER, MODEL_NAME as DEFAULT_MODEL_NAME)

# --- Configuration ---
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE_URL = "http://localhost:8000/v1"
TARGET_MODEL_NAME = DEFAULT_MODEL_NAME
LANGUAGE = "en"
CONCURRENT_REQUESTS = 256
HTTPX_TIMEOUT = httpx.Timeout(60.0, read=36000.0)
DATASET_NAME = "MLCommons/peoples_speech"
# DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"


def numpy_to_wav_bytes(waveform: np.ndarray, sampling_rate: int) -> bytes:
    """
    Converts a NumPy array of audio samples (float32) to WAV format bytes.
    """
    if waveform.dtype != np.float32:
        # Ensure waveform is float32 as expected by Whisper models.
        # If your source array has a different type, it might need normalization first.
        waveform = waveform.astype(np.float32)

    # Convert float32 to int16 for standard WAV format
    # This scales values from [-1.0, 1.0] to [-32768, 32767]
    waveform_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sampling_rate)
        wf.writeframes(waveform_int16.tobytes())
    
    return wav_buffer.getvalue()


def data_generator_openai(dataset) -> Generator[tuple[bytes, str, str, str], None, None]:
    """
    Generates audio data from dataset samples containing NumPy arrays.
    Converts NumPy arrays to WAV bytes.
    Yields:
        tuple: (audio_wav_bytes, content_type, reference_text, sample_id_for_filename)
    """
    content_type = "audio/wav" 

    for i, sample in enumerate(dataset):
        audio_data = sample.get("audio")
        reference = sample.get("text", "")
        sample_id = sample.get("id", f"sample_{i}.wav") # Use 'id' or generate one

        if not audio_data:
            print(f"Warning: 'audio' field missing or empty for sample {sample_id}. Skipping.")
            continue

        waveform_array = audio_data.get("array")
        sampling_rate = audio_data.get("sampling_rate")

        if waveform_array is None or sampling_rate is None:
            print(f"Warning: 'array' or 'sampling_rate' missing in audio data for sample {sample_id}. Skipping.")
            continue
        
        try:
            audio_wav_bytes = numpy_to_wav_bytes(waveform_array, sampling_rate)
        except Exception as e:
            print(f"Warning: Could not convert numpy array to WAV for sample {sample_id}. Error: {e}. Skipping.")
            continue
        
        # Ensure sample_id is a safe filename, replace slashes if present (common in IDs like 'path/to/file.wav')
        safe_sample_id_filename = sample_id.replace('/', '_') + ".wav" if not sample_id.lower().endswith(".wav") else sample_id.replace('/', '_')


        yield (audio_wav_bytes, content_type, reference, safe_sample_id_filename)

async def iterate_openai_response(response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Iterates over the streaming response from OpenAI API, yielding parsed chunks.
    """
    async for line in response.aiter_lines():
        if line:
            line = line.strip()
            if line.startswith("data: "):
                line_data = line[len("data: "):].strip()
                if line_data == "[DONE]":
                    break
                if not line_data: 
                    continue
                try:
                    chunk = json.loads(line_data)
                    text_segment = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if text_segment is None: 
                        text_segment = chunk.get("text")

                    if text_segment:
                        yield {
                            "time": time.perf_counter(),
                            "text_segment": text_segment
                        }
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode JSON from stream: {line_data}")
                except (KeyError, IndexError):
                    print(f"Warning: Unexpected JSON structure in stream: {chunk if 'chunk' in locals() else line_data}")
            elif line == "[DONE]":
                break


async def transcribe(
    client: httpx.AsyncClient,
    model_name: str,
    language: str,
    api_base_url: str,
    api_key: str,
    audio_bytes: bytes,
    content_type: str, 
    reference: str,
    sample_id_filename: str # Renamed from original_audio_path
) -> Dict[str, Any]:
    """
    Sends audio data (bytes) to the OpenAI-compatible transcription API and streams the response.
    """
    start_time = time.perf_counter()
    
    # sample_id_filename is already prepared to be a safe filename
    files = {"file": (sample_id_filename, audio_bytes, content_type)}
    data = {
        "model": model_name,
        "language": language,
        "stream": True,
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{api_base_url}/audio/transcriptions"

    accumulated_text = []
    chunk_arrival_times = []
    
    try:
        async with client.stream("POST", url, files=files, data=data, headers=headers) as response:
            if response.status_code == 200:
                async for result_chunk in iterate_openai_response(response):
                    accumulated_text.append(result_chunk["text_segment"])
                    chunk_arrival_times.append(result_chunk["time"])
            else:
                error_content = await response.aread()
                print(f"Error from API: {response.status_code} - {error_content.decode()} for {sample_id_filename}")
                return {
                    "start": start_time,
                    "end": time.perf_counter(),
                    "text": f"[ERROR: API returned {response.status_code}]",
                    "reference": reference,
                    "decode_times": [time.perf_counter()] if not chunk_arrival_times else chunk_arrival_times,
                }

    except httpx.RequestError as e:
        print(f"HTTPX RequestError during transcription for {sample_id_filename}: {e}")
        return {
            "start": start_time,
            "end": time.perf_counter(),
            "text": f"[ERROR: HTTPX RequestError {e}]",
            "reference": reference,
            "decode_times": [time.perf_counter()] if not chunk_arrival_times else chunk_arrival_times,
        }

    end_time = time.perf_counter()
    final_transcribed_text = "".join(accumulated_text)
    
    if not chunk_arrival_times and final_transcribed_text:
        chunk_arrival_times.append(end_time)
    elif not chunk_arrival_times:
         chunk_arrival_times.append(end_time)

    req_results = {
        "start": start_time,
        "end": end_time,
        "text": final_transcribed_text,
        "reference": reference,
        "decode_times": chunk_arrival_times,
    }
    return req_results


async def bound_transcribe(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    model_name: str,
    language: str,
    api_base_url: str,
    api_key: str,
    audio_bytes: bytes,
    content_type: str,
    reference: str,
    sample_id_filename: str # Renamed
):
    """Wraps transcribe with a semaphore to limit concurrency."""
    async with sem:
        return await transcribe(
            client, model_name, language, api_base_url, api_key, 
            audio_bytes, content_type, reference, sample_id_filename
        )

async def process_dataset_openai(
    dataset, 
    concurrent_requests: int,
    model_name: str,
    language: str,
    api_base_url: str,
    api_key: str
):
    """
    Processes the entire dataset by sending transcription requests concurrently.
    """
    tasks: List[asyncio.Task] = []
    
    sem = asyncio.Semaphore(concurrent_requests)
    print(f"Semaphore initialized with limit: {sem._value}") # _value is internal, but good for debug
    
    custom_limits = httpx.Limits(
        max_connections=1000, # e.g., if concurrent_requests is 150, this is 160
        max_keepalive_connections=1000 # Default is 20, can be increased
    )
    print(f"httpx.Limits configured: max_connections={custom_limits.max_connections}, max_keepalive_connections={custom_limits.max_keepalive_connections}")


    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, trust_env=False, limits=custom_limits) as client:
        item_count = 0
        for audio_wav_bytes, content_type, reference, sample_id_filename in data_generator_openai(dataset):
            item_count +=1
            task = asyncio.create_task(
                bound_transcribe(
                    sem, client, model_name, language, api_base_url, api_key,
                    audio_wav_bytes, content_type, reference, sample_id_filename
                )
            )
            tasks.append(task)
        
        if item_count == 0:
            print("Warning: data_generator_openai yielded no items. Check dataset structure and content.")
            return []
        
        print(f"Number of tasks created for asyncio.gather: {len(tasks)}")
        results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    print(f"Starting OpenAI Whisper streaming benchmark...")
    print(f"Model: {TARGET_MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Concurrent Requests: {CONCURRENT_REQUESTS}")
    print(f"API Base URL: {OPENAI_API_BASE_URL}")

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_hf_dataset(DATASET_NAME)
    print(f"Dataset loaded with {len(dataset)} samples.")

    if not dataset or len(dataset) == 0:
        print("Failed to load dataset or dataset is empty. Exiting.")
        exit(1)
    
    process_fn_for_eval = lambda ds, cr: process_dataset_openai(
        ds, 
        cr, 
        TARGET_MODEL_NAME, 
        LANGUAGE,
        OPENAI_API_BASE_URL,
        OPENAI_API_KEY
    )

    print("Running evaluation (including warmup)...")
    wer_score = run_evaluation(
        dataset=dataset,
        concurrent_requests=CONCURRENT_REQUESTS,
        process_dataset_fn=process_fn_for_eval,
        # n_examples=10, # Uncomment for a quick test with a subset of data
        output_dir="output_openai_whisper_array_input" # Changed output directory
    )

    print(f"Benchmark finished. Final WER: {wer_score:.2f}%")
