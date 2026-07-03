"""Triton python-backend model wrapping serve/engine.py (TurboServeEngine).

The dynamic batcher groups ragged AUDIO requests within the queue-delay window and hands
them to execute() as a list — the engine then runs them as one duration-sorted chunked
batch, exactly like the Ray backend, so the two are comparable apples-to-apples.
"""
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import json
        cfg = json.loads(args["model_config"])
        repo = cfg.get("parameters", {}).get("GRANITE_REPO", {}).get("string_value") \
            or os.environ.get("GRANITE_REPO", "/workspace/granite-speech-turbo")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from serve.engine import TurboServeEngine
        self.engine = TurboServeEngine()
        if os.environ.get("WARMUP", "1") == "1":
            self.engine.warmup()

    def execute(self, requests):
        wavs, spans = [], []
        for req in requests:
            audio = pb_utils.get_input_tensor_by_name(req, "AUDIO").as_numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            spans.append((len(wavs), 1))
            wavs.append(audio)

        texts = self.engine.transcribe_batch(wavs)

        responses = []
        for start, count in spans:
            out = pb_utils.Tensor("TEXT", np.array(
                [t.encode("utf-8") for t in texts[start:start + count]], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        self.engine = None
