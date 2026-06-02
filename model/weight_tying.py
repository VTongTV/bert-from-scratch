import torch
import torch.nn as nn


def tie_embeddings(model):
    model.embeddings.token.embedding.weight = model.cls.decoder.weight
