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

## references

devlin, j., chang, m.-w., lee, k., & toutanova, k. (2019). bert: pre-training of deep bidirectional transformers for language understanding. naacl-hlt 2019.
