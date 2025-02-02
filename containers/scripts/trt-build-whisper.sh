#! /bin/bash
set -ex

# Steps mostly from https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.16.0/docs/whisper.md
# with minor fixes to paths

wget -nc --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget -nc --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget -nc --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
# take large-v3 model as an example
wget -nc --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
# small
wget -nc --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt

source ~/scripts/trt-whisper-vars.sh
cd ~/tensorrtllm_backend/tensorrt_llm/examples/whisper/

python3 convert_checkpoint.py --model_dir ${MODEL_DIR} \
                              --output_dir ${OUTPUT_DIR} \
                              --use_weight_only \
                              --weight_only_precision $WEIGHT_ONLY_PRECISION \
                              --model_name large-v3 # or small

trtllm-build  --checkpoint_dir ${OUTPUT_DIR}/encoder \
              --output_dir ${OUTPUT_DIR}/encoder \
              --context_fmha disable \
              --moe_plugin disable \
              --gemm_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len ${MAX_SEQ_LEN} \
              --max_input_len ${MAX_INPUT_LEN} \
              --max_encoder_input_len ${MAX_ENCODER_INPUT_LEN} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}

trtllm-build  --checkpoint_dir ${OUTPUT_DIR}/decoder \
              --output_dir ${OUTPUT_DIR}/decoder \
              --max_seq_len ${MAX_SEQ_LEN} \
              --max_input_len ${MAX_INPUT_LEN} \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION} \
              --context_fmha disable

cd ~/tensorrtllm_backend
cp all_models/whisper/ model_repo_whisper -r
cp all_models/inflight_batcher_llm/tensorrt_llm model_repo_whisper -r
wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=model_repo_whisper/whisper_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

python3 tools/fill_template.py -i model_repo_whisper/tensorrt_llm/config.pbtxt triton_backend:${BACKEND},engine_dir:${DECODER_ENGINE_PATH},encoder_engine_dir:${ENCODER_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE},max_queue_size:${MAX_QUEUE_SIZE},enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACC},cross_kv_cache_fraction:${CROSS_KV_CACHE_FRACTION},encoder_input_features_data_type:TYPE_FP16
python3 tools/fill_template.py -i model_repo_whisper/whisper_bls/config.pbtxt engine_dir:${ENCODER_ENGINE_PATH},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE}

#set +x

#echo "Built model. To run Whisper with triton + tensortllm backend:"
#echo "source ~/scripts/trt-whisper-vars.sh"
#echo "cd tensorrtllm_backend/"
#echo "python3 scripts/launch_triton_server.py --world_size 1 --model_repo=model_repo_whisper/ --tensorrt_llm_model_name tensorrt_llm,whisper_bls --multimodal_gpu0_cuda_mem_pool_bytes 300000000"
