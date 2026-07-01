# bert

pytorch implementation of bert (devlin et al., 2019).

encoder-only transformer pre-trained with masked language modeling and next sentence prediction. fine-tunes on glue, squad, and swag.

## structure

- `config/` — model and training configurations
- `data/` — tokenizers, vocabulary, datasets
- `model/` — transformer encoder, heads, embeddings
- `training/` — optimizer, scheduler, training loop
- `scripts/` — pre-training, fine-tuning, ablation runners
- `eval/` — metrics and evaluators
- `tests/` — unit and integration tests

## model configs

| model | L | H | A | d_ff | params |
|-------|---|---|---|------|--------|
| bert base | 12 | 768 | 12 | 3072 | ~110m |
| bert large | 24 | 1024 | 16 | 4096 | ~340m |

## pre-training

- mlm: 15% masked, 80/10/10 replace/random/keep
- nsp: 50% isnext, 50% notnext
- loss = mean mlm + mean nsp
- adam lr 1e-4, warmup 10k steps, linear decay, batch 256

## fine-tuning

- glue: classification head W in R^{K x H}, batch 32, lr 2e-5, 3 epochs
- squad v1.1: start/end vectors, span scoring, batch 32, lr 5e-5, 3 epochs
- squad v2.0: null span at [cls], threshold tau, batch 48, lr 5e-5, 2 epochs
- swag: 4-choice, concat+dot+softmax, batch 16, lr 2e-5, 3 epochs

## ablations

- no nsp, ltr, ltr + bilstm
- model sizes: L=3/6/12, H=768/1024, A=3/12/16
- masking strategies: 80/10/10, 100% mask, 80/0/20, 80/20/0, 0/20/80, 0/0/100
- ner feature-based: concat last 4 layers, weighted sum, bilstm

## references

devlin, j., chang, m.-w., lee, k., & toutanova, k. (2019). bert: pre-training of deep bidirectional transformers for language understanding. naacl-hlt 2019.
