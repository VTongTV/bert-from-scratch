import torch
import torch.nn as nn

from config.bert_config import BertConfig
from model.bert import BertModel
from model.mlm_head import MLMPrediction, mlm_loss


class BertLTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.mlm = MLMPrediction(config.H, config.V)
        self.mlm.tie_weights(self.bert.embeddings.token.embedding)

    def forward(self, input_ids, segment_ids=None, attention_mask=None, mlm_labels=None):
        from model.attention import create_causal_mask
        if attention_mask is None:
            causal = create_causal_mask(input_ids.size(1), input_ids.device)
        else:
            causal = create_causal_mask(input_ids.size(1), input_ids.device)
            causal = causal + (1 - attention_mask.unsqueeze(1).unsqueeze(2))
        encoder_out, _ = self.bert(input_ids, segment_ids, attention_mask=attention_mask)
        mlm_logits = self.mlm(encoder_out)
        if mlm_labels is not None:
            mlm_mask = (mlm_labels != -100).float()
            mlm_labels_masked = mlm_labels.clone()
            mlm_labels_masked[mlm_labels == -100] = 0
            return mlm_loss(mlm_logits, mlm_labels_masked, mlm_mask)
        return mlm_logits
