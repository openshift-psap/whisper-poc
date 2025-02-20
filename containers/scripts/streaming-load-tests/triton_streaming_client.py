import asyncio
import queue
import sys
import time
import types

import numpy as np
import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.grpc import InferenceServerException
from tritonclient.utils import np_to_triton_dtype

from typing import List

from helpers import (data_generator,
                     run_evaluation,
                     load_hf_dataset)

CONCURRENT_REQUESTS = 64
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "D4nt3/esb-datasets-earnings22-validation-tiny-filtered"
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

async def transcribe(
    task_name: str,
    waveform: np.ndarray,
    sample_rate: int,
    reference: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    padding_duration: int = 10,
):
    task_id = int(task_name[5:])
    waveform_length = waveform.shape[0]
    duration = int(waveform_length / sample_rate)

    # padding to nearest 10 seconds
    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate *
            ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )
    samples[0, :waveform_length] = waveform
    lengths = np.array([[duration]], dtype=np.int32)

    # Prepare inputs and outputs
    inputs = [
        protocol_client.InferInput("WAV", samples.shape,
                                    np_to_triton_dtype(samples.dtype)),
        protocol_client.InferInput("WAV_LENS", lengths.shape,
                                    np_to_triton_dtype(lengths.dtype)),
        protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    input_data_numpy = np.array([WHISPER_PROMPT], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)
    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]

    # Send request
    sequence_id = 100000000 + task_id * 10


    async def async_request_iterator():
        yield {
            "model_name": "whisper_bls",
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(sequence_id)
        }
    print(f"transcribe {task_name}")
    results = []
    start = time.perf_counter()
    async for response in triton_client.stream_infer(
        inputs_iterator=async_request_iterator(),
        stream_timeout=None,
    ):
        result, error = response
        if error:
            raise ValueError
        decoding_results = result.as_numpy("TRANSCRIPTS")[0]

        if type(decoding_results) == np.ndarray:
            decoding_results = b" ".join(
                decoding_results).decode("utf-8")
        else:
            decoding_results = decoding_results.decode("utf-8")
        results.append(dict(
            time = time.perf_counter(),
            text = decoding_results
            ))
    end = time.perf_counter()
    
    req_results = dict(
        start = start,
        end = end,
        text = results[-1]["text"],
        reference = reference,
        decode_times = [res["time"] for res in results],
    )

    return req_results

async def bound_transcribe(
    sem: asyncio.Semaphore,
    task_name: str,
    waveform: np.ndarray,
    sample_rate: int,
    reference: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    padding_duration: int = 10,
):
    # Use semaphore to limit concurrent requests.
    async with sem:
        return await transcribe(
                    task_name=task_name,
                    waveform=waveform,
                    sample_rate=sample_rate,
                    reference=reference,
                    triton_client=triton_client,
                    protocol_client=protocol_client,
        )

async def process_dataset(
    dataset,
    concurrent_requests,
    base_url="localhost:8001"
):
    tasks: List[asyncio.Task] = []

    triton_client = grpcclient.InferenceServerClient(url=base_url,
                                                     verbose=False)
    protocol_client = grpcclient

    sem = asyncio.Semaphore(concurrent_requests)
    for i, (waveform, sample_rate, ref) in enumerate(data_generator(dataset)):
        task = asyncio.create_task(
            bound_transcribe(
                sem=sem,
                task_name=f"task-{i}",
                waveform=waveform,
                sample_rate=sample_rate,
                reference=ref,
                triton_client=triton_client,
                protocol_client=protocol_client,
            )
        )
        tasks.append(task)

    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    dataset = load_hf_dataset(DATASET_NAME)
    wer = run_evaluation(dataset, CONCURRENT_REQUESTS, process_dataset)
