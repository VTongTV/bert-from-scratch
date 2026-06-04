import pytest
import torch
from config.bert_config import BertConfig
from model.bert import BertModel
from model.mlm_head import MLMHead, MLMDecoder, MLMPrediction, mlm_loss


def test_mlm_head():
    H = 64
    head = MLMHead(H)
    x = torch.randn(2, 8, H)
    out = head(x)
    assert out.shape == (2, 8, H)


def test_mlm_decoder():
    H, V = 64, 100
    decoder = MLMDecoder(H, V)
    x = torch.randn(2, 8, H)
    out = decoder(x)
    assert out.shape == (2, 8, V)


def test_mlm_loss():
    V = 100
    logits = torch.randn(2, 8, V)
    labels = torch.randint(0, V, (2, 8))
    mask = torch.ones(2, 8)
    mask[0, 5:] = 0
    loss = mlm_loss(logits, labels, mask)
    assert loss.item() > 0
    assert loss.ndim == 0


def test_mlm_prediction():
    H, V = 64, 100
    pred = MLMPrediction(H, V)
    x = torch.randn(2, 8, H)
    out = pred(x)
    assert out.shape == (2, 8, V)


def test_mlm_weight_tying():
    H, V = 64, 100
    pred = MLMPrediction(H, V)
    emb = torch.nn.Embedding(V, H)
    pred.tie_weights(emb)
    assert pred.decoder.decoder.weight is emb.weight


def test_mlm_integration_with_bert():
    config = BertConfig(L=2, H=64, A=4, V=100, max_len=32)
    model = BertModel(config)
    mlm = MLMPrediction(config.H, config.V)
    mlm.tie_weights(model.embeddings.token.embedding)
    input_ids = torch.randint(0, 100, (2, 8))
    encoder_out, _ = model(input_ids)
    logits = mlm(encoder_out)
    assert logits.shape == (2, 8, config.V)
    labels = torch.randint(0, config.V, (2, 8))
    mask = torch.ones(2, 8)
    loss = mlm_loss(logits, labels, mask)
    assert loss.item() > 0
