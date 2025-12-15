from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments, Trainer
from transformers import (
    AutoModel, 
    AutoConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = None

        if isinstance(alpha, torch.Tensor) and len(alpha) > 1:
            self.alpha = torch.stack([
                torch.ones_like(alpha),
                alpha,
            ]).T
        else:
            self.alpha = alpha

        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        alpha = None

        if isinstance(alpha, torch.Tensor) and len(alpha) > 1:
            alpha =  torch.gather(
                self.alpha.repeat(targets.shape[0], 1, 1),
                dim=2,
                index=targets.to(torch.long).unsqueeze(2)
            ).squeeze(2)
        else:
            alpha = self.alpha

        focal_loss = alpha * (1 - torch.exp(-bce)) ** self.gamma * bce
        return focal_loss.mean()

@dataclass
class PolarTrainingArgs(TrainingArguments):
    layer_wise: bool = field(
        default=False,
        metadata={"help": "Enable layer-wise learning rate decay"}
    )
    layer_wise_decay: float = field(
        default=0.95,
        metadata={"help": "Layer-wise learning rate decay factor"}
    )
    classifier_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for classifier head"}
    )


class PolarModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels, hidden_layers, criterion="bce", pooling_strategy="cls", weights=None, alpha=1, gamma=2):
        super(PolarModel, self).__init__()

        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy.lower()

        self.criterion = None

        if criterion.lower() == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        elif criterion.lower() == "focal":
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


        self.config = AutoConfig.from_pretrained(checkpoint)
        self.base_model = AutoModel.from_pretrained(checkpoint, config=self.config)

        input_size = self.config.hidden_size
        layers = []

        for hidden_size in hidden_layers:
            layers.extend([
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3)
            ])
            input_size = hidden_size

        layers.extend([
            torch.nn.Dropout(0.3),
            torch.nn.Linear(input_size, self.num_labels)
        ])

        self.classifier = torch.nn.Sequential(*layers)

    def _pool_output(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self._pool_output(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PolarTrainer(Trainer):

    def _get_encoder_layers(self, model):
        base_model = model.base_model
        
        if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
            return base_model.encoder.layer
        elif hasattr(base_model, 'layer'):
            return base_model.layer
        elif hasattr(base_model, 'layers'):
            return base_model.layers
        else:
            raise ValueError("Could not identify encoder layers in base model")

    def _get_embeddings(self, model):
        base_model = model.base_model
        
        if hasattr(base_model, 'embeddings'):
            return base_model.embeddings
        elif hasattr(base_model, 'embed_tokens'):
            return base_model.embed_tokens
        else:
            raise ValueError("Could not identify embeddings in base model")

    def _build_layer_wise_param_groups(self):
        model = self.model
        args = self.args

        try:
            layers = self._get_encoder_layers(model)
            embeddings = self._get_embeddings(model)
        except ValueError as e:
            raise ValueError(f"Layer-wise learning rates failed: {e}. Disable layer_wise or check model architecture.")

        n_layers = len(layers)
        base_lr = args.learning_rate
        lr_decay = args.layer_wise_decay
        classifier_lr = args.classifier_lr

        param_groups = [{
            'params': embeddings.parameters(),
            'lr': base_lr * (lr_decay ** n_layers),
        }]

        for depth, layer in enumerate(layers):
            decayed_lr = base_lr * (lr_decay ** (n_layers - depth - 1))
            param_groups.append({
                'params': layer.parameters(),
                'lr': decayed_lr,
            })

        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': classifier_lr,
        })

        return param_groups

    def create_optimizer(self):
        if self.optimizer is None:
            if isinstance(self.args, PolarTrainingArgs) and self.args.layer_wise:
                param_groups = self._build_layer_wise_param_groups()
            else:
                decay_params = []
                no_decay_params = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if any(nd in name for nd in ["bias", "LayerNorm", "layer_norm"]):
                            no_decay_params.append(param)
                        else:
                            decay_params.append(param)
                
                param_groups = [
                    {'params': decay_params, 'weight_decay': self.args.weight_decay},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]

            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        return self.optimizer