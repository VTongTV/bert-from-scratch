# agents.md — bert implementation

## project

implement devlin et al. (2019), arxiv:1810.04805v2. bert: pre-training of deep bidirectional transformers for language understanding. encoder-only transformer with masked lm and next sentence prediction. pytorch. ground truth is the paper tex files in `arXiv-1810.04805v2/`.

## code style

- write the smallest, simplest code. this is not production. let it fail.
- no unnecessary error handling. no try/except unless the paper specifies a numeric boundary.
- no type hints. no abstract base classes. no factories.
- docstrings: one or two sentences max. only when you write a function. try to avoid them.
- no comments in code. the code is the comment. except: `# section 3.1` when a formula needs a pointer to the paper.
- imports: stdlib, then torch, then local. one per line.

## language

follow `D:\Implemented papers\SimpleEnglish` everywhere. rules:

- lowercase, informal, concise. no filler.
- commit messages: one short paragraph. lowercase. no period at end.
- code comments: none unless a formula needs a pointer to the paper. then: `# eq 1` or `# section 3.4`.
- variable names: match the paper. `L`, `H`, `A`, `d_ff`, `P_drop`. no synonyms. use `L` for layers, `H` for hidden size, `A` for attention heads, not `num_layers`, `hidden_size`, `num_heads`.
- no "we", "our", "note that", "it is worth noting", "simply", "just", "easily".
- one word = one meaning. pick "loss" or "cost" and keep it. pick "mask" or "attention_mask" and keep it.

## git

- all commits to `main`. no branches. no merges.
- maximize commit count. one logical change per commit.
- commit message format: `<scope>: <what changed>`. scope is one of: `init`, `config`, `data`, `model`, `training`, `scripts`, `eval`, `ablation`, `docs`.
- commit message is one short paragraph. lowercase. no trailing period. imperative verb first in the subject.
- optional body: one to three sentences. simple tense. active voice. state what changed and how it maps to the paper. max 25 words per sentence.
- use exact dates from `timetable.md`. set both author and committer dates for each commit. you can add intermediate commits to fill the timeline but do not skip or reorder the listed ones.
- `.gitignore` all non-implementation files: paper pdf, paper md, timetable, agents.md, initiation prompts, plan files, datasets, checkpoints. the repo history must look like a clean implementation project.
- do not mention download scripts or data fetchers in commit messages. include the code but keep the commits silent about it.

## commit audit checks

run these before each commit. fix every hit.
1. subject scope is in the allowed list.
2. subject is lowercase, imperative, no trailing period, no filler.
3. body is one paragraph, at most three sentences, each at most 25 words.
4. no contractions: 'll, 're, 's, n't, it's.
5. no perfect tenses: has been, have been, had been.
6. no banned modals: should, would, may, might, could.
7. no "-ing" clauses after a comma: , making, , allowing, , enabling.
8. no semicolons.
9. no latin abbreviations: e.g., i.e., etc.
10. no filler: simply, easily, seamlessly, robust, leverage, utilize, comprehensive.
11. no mention of dataset or downloader scripts.
12. commit date matches the timetable day.

## paper specifics

these come from the tex files. verify every constant against the paper:

- bert base: L=12, H=768, A=12, 110m params. bert large: L=24, H=1024, A=16, 340m params
- feed-forward size is 4H (3072 for H=768, 4096 for H=1024)
- wordpiece vocab of 30,000. max sequence length 512
- input = token + segment + position embeddings. [cls] first, [sep] between pairs
- mlm: mask 15% of wordpiece tokens. 80% replace with [mask], 10% random token, 10% unchanged
- nsp: 50% isnext, 50% notnext. pooler is tanh on [cls]
- pre-training: batch 256, 1m steps, adam lr 1e-4, b1=0.9, b2=0.999, l2 decay 0.01, warmup 10k steps, linear decay, dropout 0.1, gelu. loss = mean mlm + mean nsp. seq len 128 for 90% of steps, 512 for 10%
- fine-tuning: batch 16 or 32, lr 5e-5/3e-5/2e-5, epochs 2/3/4, dropout 0.1
- glue: classification head W in R^{K x H}, loss = log(softmax(C W^T)). batch 32, 3 epochs, lr search over 5e-5/4e-5/3e-5/2e-5. random restarts for bert large
- squad v1.1: start vector S, end vector E. P_i = softmax(S . T_i). span score S . T_i + E . T_j, max over j >= i. 3 epochs, lr 5e-5, batch 32
- squad v2.0: no-answer span at [cls]. s_null = S . C + E . C. predict non-null if best span > s_null + tau. 2 epochs, lr 5e-5, batch 48
- swag: 4 choices, concat sentence a + continuation b, vector dot [cls], softmax. 3 epochs, lr 2e-5, batch 16
- ablations: no nsp, ltr and no nsp, ltr + bilstm. model sizes L=3/6/12, H=768/1024, A=3/12/16. masking 80/10/10, 100% mask, 80/0/20, 80/20/0, 0/20/80, 0/0/100. ner feature-based: concat last 4 layers, weighted sum, bilstm

## agent rules

- you are the sole implementer. no subagents except oracle.
- oracle can read tex/pdf files when you need to verify a detail against the paper.
- before you write any component, read the relevant paper section first.
- after you write any component, read the paper section again and verify every constant.
- test immediately after each component. a test commit follows its component commit on the same day.
- do not skip rest days. do not commit on rest days.
- do not move to the next phase until the current phase tests pass.

## directory structure

```
config/         # model and training configs
data/           # tokenizers, vocab, datasets, dataloaders
model/          # all nn.module classes
train/          # optimizer, scheduler, loss, training loop
scripts/        # pre-training, glue, squad, swag, ablation runners
eval/           # metrics, evaluators
test/           # integration tests
```

## test standard

- pytest. one test file per module.
- test file naming: `test_<module>.py`.
- test what the paper specifies: output shapes, constant values, masking correctness, scaling behavior.
- do not test pytorch internals or framework behavior.
