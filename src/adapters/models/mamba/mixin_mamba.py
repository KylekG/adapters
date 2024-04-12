from torch import nn
from adapters.methods.prefix_tuning import PrefixTuningLayer
from ...methods.lora import LoRALinear
from ...methods.bottleneck import BottleneckLayer
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import ModelBaseAdaptersMixin, EmbeddingAdaptersMixin, InvertibleAdaptersMixin
from ...composition import adjust_tensors_for_parallel_

# class MambaBlockAdaptersMixin:
#     def init_adapters(self, model_config, adapters_config):


# Each MambaMixer corresponds to a Layer in the architecture:
class MambaMixerAdapterMixin:
    """Adds adapters to the MambaMixer module."""

    def init_adapters(self, model_config, adapters_config):

        # self.use_conv_bias = model_config.use_conv_bias

        # Set the location_key to selfattn (HACK, and just in case it hasn't been set), that allows for
        # configuration of the LoRA Modules using the attn_keys of
        # the adaptable components:
        self.location_key = "selfattn"
        self.in_proj = LoRALinear.wrap(
            module=self.in_proj,
            location_key=self.location_key,
            attn_key="in_proj",
            model_config=model_config,
            adapters_config=adapters_config,
            bias=False
        )
        self.x_proj = LoRALinear.wrap(
            module=self.x_proj,
            location_key=self.location_key,
            attn_key="x_proj",
            model_config=model_config,
            adapters_config=adapters_config,
            bias=False
        )
        self.dt_proj = LoRALinear.wrap(
            module=self.dt_proj,
            location_key=self.location_key,
            attn_key="dt_proj",
            model_config=model_config,
            adapters_config=adapters_config,
            bias=True
        )
        self.out_proj = LoRALinear.wrap(
            module=self.out_proj,
            location_key=self.location_key,
            attn_key="out_proj",
            model_config=model_config,
            adapters_config=adapters_config,
            bias=False
        )


class MambaBlockAdapterMixin(BottleneckLayer):
    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)
        # self.prefix_tuning = PrefixTuningLayer(
        #     location_key="encoder_prefix",
        #     model_config=model_config,
        #     adapters_config=adapters_config,
        #     add_model_type_to_key=False
        # )


class MambaModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config, add_prefix_tuning_pool=False)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        # Register hook for post embedding forward:
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self):
        for i, layer in enumerate(self.layers):
            yield i, layer
