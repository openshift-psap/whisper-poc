#! /bin/bash
set -ex

# Once the container is running execute:
source ~/scripts/trt-whisper-vars.sh
cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper
python3 run.py --engine_dir $OUTPUT_DIR \
               --dataset hf-internal-testing/librispeech_asr_dummy \
               --enable_warmup \
               --name librispeech_dummy_large_v3 \
               --assets_dir ~/assets \
               --batch_size ${MAX_BATCH_SIZE} \
               --num_beams ${MAX_BEAM_WIDTH}
