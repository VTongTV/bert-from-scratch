import torch
import torch.nn as nn

from model.bert import BertModel
from model.mlm_head import MLMPrediction, mlm_loss
from model.nsp_head import NSPHead, nsp_loss


class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.mlm = MLMPrediction(config.H, config.V)
        self.nsp = NSPHead(config.H)
        self.mlm.tie_weights(self.bert.embeddings.token.embedding)

    def forward(self, input_ids, segment_ids=None, attention_mask=None, mlm_labels=None, nsp_labels=None):
        encoder_out, pooled_out = self.bert(input_ids, segment_ids, attention_mask)
        mlm_logits = self.mlm(encoder_out)
        nsp_logits = self.nsp(pooled_out)
        if mlm_labels is not None and nsp_labels is not None:
            mlm_mask = (mlm_labels != -100).float()
            mlm_labels_masked = mlm_labels.clone()
            mlm_labels_masked[mlm_labels == -100] = 0
            loss_mlm = mlm_loss(mlm_logits, mlm_labels_masked, mlm_mask)
            loss_nsp = nsp_loss(nsp_logits, nsp_labels)
            return loss_mlm + loss_nsp
        return mlm_logits, nsp_logits
