
from typing import Optional, Dict
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.mamba.modeling_mamba import MAMBA_START_DOCSTRING, MAMBA_INPUTS_DOCSTRING, MambaModel, MambaPreTrainedModel
from ...context import AdapterSetup
from ...heads import ModelWithFlexibleHeadsAdaptersMixin

from ...model_mixin import EmbeddingAdaptersWrapperMixin

from ...wrappers import init

# from ...configuration import ModelAdaptersConfig
from .heads_mamba import MambaSequenceClassificationHead

# from ...context import AdapterSetup
# from ...heads import ModelWithFlexibleHeadsAdaptersMixin
# from ...model_mixin import EmbeddingAdaptersWrapperMixin
# from ...wrappers import init


@add_start_docstrings(
    """A Pretrained Mamba Model with the option to add multiple flexible heads on top.""",
    MAMBA_START_DOCSTRING,
)
class MambaAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, MambaPreTrainedModel):
    head_types = ["causal_lm", "classification"]

    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__(config)

        self.backbone = MambaModel(config)

        init(self.backbone)

        self.register_custom_head(
            identifier="MambaSequenceClassificationHead",
            head=MambaSequenceClassificationHead
        )

        self._init_head_modules()

        self.init_weights()

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        inference_params=None,
        **kwargs
    ):

        attention_mask = attention_mask.view(
            -1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)
                                         ) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2),
                               inputs_embeds.size(-1)) if inputs_embeds is not None else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, context = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
            inference_params=inference_params,
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context
        head_inputs = outputs

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def add_classification_head(
        self,
        head_name,
        multilabel=False,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
        dropout_prob=None
    ):

        if multilabel:
            assert False, "Unimplemented"
        else:
            self.add_custom_head(
                head_type="MambaSequenceClassificationHead",
                head_name=head_name,
                num_labels=num_labels,
                layers=layers,
                activation_function=activation_function,
                id2label=id2label,
                use_pooler=use_pooler,
                bias=bias,
                dropout_prob=dropout_prob,
            )
