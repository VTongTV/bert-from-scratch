import torch
import torch.nn as nn

from model.pretrain_model import BertForPreTraining


class BertForPreTrainingNoNSP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_for_pretraining = BertForPreTraining(config)

    def forward(self, input_ids, segment_ids=None, attention_mask=None, mlm_labels=None, nsp_labels=None):
        mlm_logits, nsp_logits = self.bert_for_pretraining(input_ids, segment_ids, attention_mask)
        if mlm_labels is not None:
            from model.mlm_head import mlm_loss
            mlm_mask = (mlm_labels != -100).float()
            mlm_labels_masked = mlm_labels.clone()
            mlm_labels_masked[mlm_labels == -100] = 0
            return mlm_loss(mlm_logits, mlm_labels_masked, mlm_mask)
        return mlm_logits, nsp_logits
