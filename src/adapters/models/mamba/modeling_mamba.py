from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.mamba.modeling_mamba import MambaMixer, MambaCache, MambaBlock


from ...composition import adjust_tensors_for_parallel
from .mixin_mamba import MambaBlockAdapterMixin, MambaMixerAdapterMixin

from transformers.utils.import_utils import is_mamba_ssm_available, is_causal_conv1d_available
if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn,
     causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


class MambaMixerWithAdapters(MambaMixerAdapterMixin, MambaMixer):
    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states=hidden_states, cache_params=cache_params)
        return self.slow_forward(input_states=hidden_states, cache_params=cache_params)


class MambaBlockWithAdapters(MambaBlockAdapterMixin, MambaBlock):

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        residual = hidden_states

        hidden_states = self.norm(
            hidden_states.to(dtype=self.norm.weight.dtype)
        )

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states=hidden_states, cache_params=cache_params)
        # hidden_states = residual + hidden_states
        # return hidden_states
        return self.bottleneck_layer_forward(hidden_states, residual, None)
