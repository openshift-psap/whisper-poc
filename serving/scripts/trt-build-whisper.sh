#! /bin/bash
set -ex

source ~/scripts/trt-whisper-vars.sh
cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper/

echo "Converting the checkpoints"
python3 convert_checkpoint.py --model_dir ${MODEL_DIR} \
                              --output_dir ${OUTPUT_DIR} \
                              --model_name ${MODEL_NAME}

echo "Building the encoder"
# Encoder build
trtllm-build --checkpoint_dir ${OUTPUT_DIR}/encoder \
             --output_dir ${OUTPUT_DIR}/encoder \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --gemm_plugin disable \
             --max_input_len ${MAX_ENCODER_INPUT_LEN} \
             --max_seq_len ${MAX_ENCODER_SEQ_LEN} \
             --log_level error

echo "Building the decoder"
# Decoder build
trtllm-build --checkpoint_dir ${OUTPUT_DIR}/decoder \
             --output_dir ${OUTPUT_DIR}/decoder \
             --max_beam_width ${MAX_BEAM_WIDTH} \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --max_seq_len ${MAX_DECODER_SEQ_LEN} \
             --max_input_len ${MAX_DECODER_INPUT_LEN} \
             --max_encoder_input_len ${MAX_ENCODER_INPUT_LEN} \
             --gemm_plugin ${INFERENCE_PRECISION} \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --gpt_attention_plugin ${INFERENCE_PRECISION} \
             --log_level error

cd ~/tensorrtllm_backend
cp all_models/whisper/ model_repo_whisper -r
cp all_models/inflight_batcher_llm/tensorrt_llm model_repo_whisper -r
wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

echo "Running the fill template script"
python3 tools/fill_template.py -i model_repo_whisper/tensorrt_llm/config.pbtxt triton_backend:${BACKEND},engine_dir:${DECODER_ENGINE_PATH},encoder_engine_dir:${ENCODER_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE},enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACC},cross_kv_cache_fraction:${CROSS_KV_CACHE_FRACTION},encoder_input_features_data_type:TYPE_FP16
python3 tools/fill_template.py -i model_repo_whisper/whisper_bls/config.pbtxt engine_dir:${ENCODER_ENGINE_PATH},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE}

#set +x

#echo "Built model. To run Whisper with triton + tensortllm backend:"
#echo "source ~/scripts/trt-whisper-vars.sh"
#echo "cd tensorrtllm_backend/"
#echo "python3 scripts/launch_triton_server.py \
#      --world_size 1 \
#      --model_repo=model_repo_whisper/ \
#      --tensorrt_llm_model_name tensorrt_llm,whisper_bls \
#      --multimodal_gpu0_cuda_mem_pool_bytes 300000000"

# cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper

# cd
# # Once the container is running execute:
# source ~/scripts/trt-whisper-vars.sh
# # cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper
# python3 ~/scripts/run_trt.py --engine_dir $OUTPUT_DIR \
#                --dataset hf-internal-testing/librispeech_asr_dummy \
#                --enable_warmup \
#                --name librispeech_dummy_large_v3 \
#                --assets_dir ~/assets \
#                --num_beams ${MAX_BEAM_WIDTH}
