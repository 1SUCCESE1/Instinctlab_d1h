"""DDT ONNX teacher wrapper for TPPO distillation.

Wraps a DDT flat.onnx model (onnxruntime) with a 10-frame history buffer to match
the deployment rl_controller's input format (current obs [33] + 10-frame history [10×33]).
Implements the instinct_rl TPPO teacher interface: act(), act_inference(), reset().
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxTeacher(nn.Module):
    """DDT ONNX model wrapped as a torch-compatible teacher for TPPO.

    The DDT rl_controller feeds flat.onnx with two inputs:
        nn_input0: (N, 33)  — current frame observation
        nn_input1: (N, 10, 33) — 10-frame history (current obs excluded)
    This wrapper maintains the history buffer and batches the onnx call.
    """

    def __init__(self, onnx_path: str, obs_dim: int = 33, history_len: int = 10, **kwargs):
        if ort is None:
            raise ImportError("onnxruntime is required for OnnxTeacher. Install with: pip install onnxruntime")
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._history: np.ndarray | None = None
        # Identify input names from the model
        inputs = self.onnx_session.get_inputs()
        self._obs_input_name = inputs[0].name      # nn_input0
        self._hist_input_name = inputs[1].name     # nn_input1

    def init_history(self, num_envs: int):
        """Allocate history buffer. Call once after env creation."""
        self._history = np.zeros((num_envs, self.history_len, self.obs_dim), dtype=np.float32)

    def reset(self, done_ids):
        """Zero history for envs that just reset. Called by TPPO.process_env_step."""
        if self._history is not None and len(done_ids) > 0:
            self._history[done_ids] = 0.0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Batched inference: obs (N, 33) → actions (N, 8)."""
        if self._history is None:
            self.init_history(obs.shape[0])
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        hist_np = self._history.copy()
        inputs = {self._obs_input_name: obs_np, self._hist_input_name: hist_np}
        actions = self.onnx_session.run(None, inputs)[0]  # (N, 8)
        # shift history and append current obs
        self._history = np.roll(self._history, -1, axis=1)
        self._history[:, -1, :] = obs_np
        return torch.from_numpy(actions).to(obs.device).detach()

    def act(self, obs):
        return self.forward(obs)

    def act_inference(self, obs):
        return self.forward(obs)

    def eval(self):
        return self
