from transformers.models.mamba.modeling_mamba import MambaSequenceClassificationOutput
    
from transformers.modeling_outputs import Seq2SeqModelOutput
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from typing import Optional, Dict

from ...heads import PredictionHead


class MambaSequenceClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "MambaSequenceClassificationHead",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            cls_output = self._get_cls_output(outputs, **kwargs)
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                assert False, "Unimplemented"
            else:
                return MambaSequenceClassificationOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    cache_params=outputs.cache_params,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs
