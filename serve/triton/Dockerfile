# Triton python-backend image for granite-speech-turbo
#
#   docker build -t granite-asr-triton -f serve/triton/Dockerfile .
#   docker run --gpus 1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#     -v /path/to/model_snapshot:/models/granite -e MODEL_DIR=/models/granite \
#     -v inductor-cache:/root/.inductor -e TORCHINDUCTOR_CACHE_DIR=/root/.inductor \
#     granite-asr-triton
#
# The inductor-cache volume persists torch.compile artifacts: first boot pays the
# 5-10 min autotune once, every restart after that comes up hot.
FROM nvcr.io/nvidia/tritonserver:25.05-py3

# proven stack: torch 2.9.1 (cu128 index still carries it; cu130 index does not)
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^torch==' /tmp/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Triton-image environment repairs (all hit in practice on 25.05-py3):
#  * the DALI backend's bundled wheel dir shadows attrs/packaging on the interactive
#    PYTHONPATH, so pip thinks they're installed while non-login shells can't import them
#  * the image ships protobuf 7.x; ray-serve needs the pre-6 FieldDescriptor API
RUN pip install --no-cache-dir --force-reinstall --no-deps attrs packaging wheel wrapt && \
    pip install --no-cache-dir --force-reinstall "protobuf~=5.29.0"

COPY . /workspace/granite-speech-turbo
ENV GRANITE_REPO=/workspace/granite-speech-turbo
ENV GRANITE_PTX_CACHE=/root/.ptx

CMD ["tritonserver", "--model-repository=/workspace/granite-speech-turbo/serve/triton/model_repository"]
