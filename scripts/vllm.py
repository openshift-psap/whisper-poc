from datasets import load_dataset
from vllm import LLM, SamplingParams
import time

dataset = load_dataset("MLCommons/peoples_speech", "validation")
dataset = dataset.filter(lambda example: example['duration_ms'] < 30000 and example['duration_ms'] > 10000)
data_subset = dataset["validation"]

tensor_parallel_size = 1
model = "openai/whisper-large-v3" #["openai/whisper-small", "openai/whisper-large-v3-turbo"]
max_num_seqs = 100
gpu_memory_utilization=0.95

llm = LLM(
    model=model,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    enforce_eager=True,
    max_num_seqs=max_num_seqs
)
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=200,
)

prompt_batch = [
        {
        "prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": (sample["audio"]["array"], sample["audio"]["sampling_rate"])
        }
    }
        for sample in data_subset
    ]

total_audio_seconds = sum([sample["duration_ms"] for sample in data_subset]) / 1000

expected_output = [sample["text"] for sample in data_subset]

start_time = time.time()

outputs = llm.generate(
        prompt_batch,
        sampling_params)

end_time = time.time()
total_time = end_time - start_time

for output, expected in zip(outputs, expected_output):
    print(f"expected: {expected}")
    print(f"output: {output.outputs[0].text}")


print(f"Elapsed time: {total_time}")

print(f"Total audio seconds processed: {total_audio_seconds}")
seconds_transcribed_per_sec = total_audio_seconds / total_time
print(f"Seconds transcribed / sec: {seconds_transcribed_per_sec}")

rps = len(data_subset) / total_time
print(f"Requests per second: {rps} for {len(data_subset)}")
