"""DDT ONNX teacher wrapper for TPPO distillation.

Re-implements the DDT flat.onnx network in pure PyTorch for batched GPU inference.
The original onnx model uses fixed batch=1 and has Slice ops incompatible with
batched onnx2torch conversion. This module loads the weights from the onnx file
and provides a hand-built torch equivalent that supports any batch size.

DDT flat.onnx architecture:
  Encoder: history(10×33→330) → LN(256) → LN(128) → embed(26)
  Actor:   concat(obs(33), embed(26)) → actions(8)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDTPolicy(nn.Module):
    """DDT flat.onnx re-implemented as a torch module for batched GPU inference."""

    def __init__(self, onnx_path: str):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        onnx_model = onnx.load(onnx_path)
        W = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()).float()
             for i in onnx_model.graph.initializer}

        def _load(dst, src_key):
            return nn.Parameter(W[src_key])

        # Encoder: 330 → 512 → 256 → LN(256) → 128 → LN(128) → 26
        self.enc_fc1 = nn.Linear(330, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_ln1 = nn.LayerNorm(256)
        self.enc_fc3 = nn.Linear(256, 128)
        self.enc_ln2 = nn.LayerNorm(128)
        self.enc_fc4 = nn.Linear(128, 26)

        self.enc_fc1.weight = _load(self.enc_fc1.weight, 'mlp_encoder.0.weight')
        self.enc_fc1.bias   = _load(self.enc_fc1.bias,   'mlp_encoder.0.bias')
        self.enc_fc2.weight = _load(self.enc_fc2.weight, 'mlp_encoder.2.weight')
        self.enc_fc2.bias   = _load(self.enc_fc2.bias,   'mlp_encoder.2.bias')
        self.enc_ln1.weight = _load(self.enc_ln1.weight, 'mlp_encoder.4.weight')
        self.enc_ln1.bias   = _load(self.enc_ln1.bias,   'mlp_encoder.4.bias')
        self.enc_fc3.weight = _load(self.enc_fc3.weight, 'mlp_encoder.5.weight')
        self.enc_fc3.bias   = _load(self.enc_fc3.bias,   'mlp_encoder.5.bias')
        self.enc_ln2.weight = _load(self.enc_ln2.weight, 'mlp_encoder.7.weight')
        self.enc_ln2.bias   = _load(self.enc_ln2.bias,   'mlp_encoder.7.bias')
        self.enc_fc4.weight = _load(self.enc_fc4.weight, 'mlp_encoder.8.weight')
        self.enc_fc4.bias   = _load(self.enc_fc4.bias,   'mlp_encoder.8.bias')

        # Actor: 59 → 512 → 256 → 128 → 8
        self.act_fc1 = nn.Linear(59, 512)
        self.act_fc2 = nn.Linear(512, 256)
        self.act_fc3 = nn.Linear(256, 128)
        self.act_fc4 = nn.Linear(128, 8)

        self.act_fc1.weight = _load(self.act_fc1.weight, 'actor.0.weight')
        self.act_fc1.bias   = _load(self.act_fc1.bias,   'actor.0.bias')
        self.act_fc2.weight = _load(self.act_fc2.weight, 'actor.2.weight')
        self.act_fc2.bias   = _load(self.act_fc2.bias,   'actor.2.bias')
        self.act_fc3.weight = _load(self.act_fc3.weight, 'actor.4.weight')
        self.act_fc3.bias   = _load(self.act_fc3.bias,   'actor.4.bias')
        self.act_fc4.weight = _load(self.act_fc4.weight, 'actor.6.weight')
        self.act_fc4.bias   = _load(self.act_fc4.bias,   'actor.6.bias')

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (N, 33) current observation
            history: (N, 10, 33) past 10 frames of observations
        Returns:
            actions: (N, 8) mean actions
        """
        flat_hist = history.reshape(obs.size(0), -1)  # (N, 330)

        # Encoder
        x = F.elu(self.enc_fc1(flat_hist))
        x = F.elu(self.enc_fc2(x))
        x = self.enc_ln1(x)
        x = F.elu(self.enc_fc3(x))
        x = self.enc_ln2(x)
        embed = self.enc_fc4(x)  # (N, 26)

        # Actor
        x = torch.cat([obs, embed], dim=1)  # (N, 59)
        x = F.elu(self.act_fc1(x))
        x = F.elu(self.act_fc2(x))
        x = F.elu(self.act_fc3(x))
        return self.act_fc4(x)  # (N, 8)


class OnnxTeacher(nn.Module):
    """DDT teacher wrapped as a torch-compatible teacher for TPPO.

    Maintains a 10-frame history buffer. Accepts obs (N, 33), runs DDTPolicy batched,
    returns actions (N, 8). Compatible with instinct_rl TPPO's act/act_inference/reset.
    """

    def __init__(self, onnx_path: str, obs_dim: int = 33, history_len: int = 10, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.model = DDTPolicy(onnx_path).eval()
        self._history: torch.Tensor | None = None

    def init_history(self, num_envs: int, device: torch.device | None = None):
        """Allocate history buffer and move model to device."""
        if device is None:
            device = next(self.model.parameters()).device
        self._history = torch.zeros(num_envs, self.history_len, self.obs_dim, device=device)
        self.model.to(device)

    def reset(self, done_ids):
        """Zero history for envs that just reset. Called by TPPO.process_env_step."""
        if self._history is not None and len(done_ids) > 0:
            self._history[done_ids] = 0.0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Batched GPU inference: obs (N, 33) → actions (N, 8)."""
        device = obs.device
        if self._history is None or self._history.device != device:
            self.init_history(obs.shape[0], device)
        self._history = torch.roll(self._history, -1, dims=1)
        self._history[:, -1, :] = obs
        with torch.no_grad():
            return self.model(obs, self._history)

    def act(self, obs):
        return self.forward(obs)

    def act_inference(self, obs):
        return self.forward(obs)

    def eval(self):
        self.model.eval()
        return self
