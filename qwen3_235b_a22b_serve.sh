source /workspace/zhangzitian/miniforge3/bin/activate verl050

# 机器之间通过内网 ip 访问
hostname -I  

# 定义模型名称。
MODEL_NAME="qwen3-235b-a22b"

# 定义服务运行时监听的端口号。可以根据实际需求进行调整，默认使用30000端口
PORT="8001"

# 定义使用的GPU数量。这取决于实例上可用的GPU数量，可以通过nvidia-smi -L命令查询
TENSOR_PARALLEL_SIZE="8"

# 设置本地存储路径
LOCAL_SAVE_PATH="/sharedata/zhangzitian/models/Qwen3-235B-A22B"

#     --max-model-len=16384 \
vllm serve ${LOCAL_SAVE_PATH} --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --trust-remote-code \
    --host 0.0.0.0 --port ${PORT} --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill --enable-expert-parallel \
    --enable-reasoning --reasoning-parser deepseek_r1
