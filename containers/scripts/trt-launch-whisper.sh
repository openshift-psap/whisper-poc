set +x

source ~/scripts/trt-whisper-vars.sh
cd tensorrtllm_backend/
python3 scripts/launch_triton_server.py \
     --world_size 1 \
     --model_repo=model_repo_whisper/ \
     --tensorrt_llm_model_name tensorrt_llm,whisper_bls \
     --multimodal_gpu0_cuda_mem_pool_bytes 300000000
