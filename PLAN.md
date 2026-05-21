# plan — bert implementation

this plan maps the rebuild of bert (devlin et al., 2019) to the commit
schedule in `timetable.md`. the implementation agent follows this plan and
obeys `agents.md`. the paper in `arXiv-1810.04805v2/` is the ground truth.

## summary

| item | value |
|---|---|
| paper | bert, devlin et al., 2019 (naacl-hlt) |
| framework | pytorch |
| start | may 21, 2026 |
| end | july 1, 2026 |
| working days | 36 |
| rest days | 6 |
| total commits | 360 |
| branch | main only |
| commit dates | exact, from timetable.md |

## phases

| phase | dates | working | rest | commits | focus |
|---|---|---|---|---|---|
| 1 | may 21 to may 27 | 6 | 1 | 60 | scaffolding and data pipeline |
| 2 | may 28 to jun 3 | 6 | 1 | 60 | transformer architecture |
| 3 | jun 4 to jun 10 | 6 | 1 | 60 | pre-training |
| 4 | jun 11 to jun 17 | 6 | 1 | 60 | fine-tuning glue |
| 5 | jun 18 to jun 24 | 6 | 1 | 60 | fine-tuning qa and swag |
| 6 | jun 25 to jul 1 | 6 | 1 | 60 | ablations and analysis |

rest days (no commits): may 24, jun 1, jun 7, jun 14, jun 21, jun 28.

## phase 1 — scaffolding and data pipeline (may 21 to may 27, commits 1 to 60)

goal: repo skeleton, configs, tokenizer, vocabulary, input embeddings, and all
data loaders for pre-training, glue, squad, and swag.

- day 1 (may 21): init repo, readme, requirements, setup, gitignore, bert
  config dataclass, base preset, large preset, pre-training config, fine-tuning
  config. commits 1 to 10.
- day 2 (may 22): special tokens, vocabulary, wordpiece tokenizer, greedy
  longest-match, ## prefix, encode, encode_pair, decode, vocab builder, tests.
  commits 11 to 20.
- day 3 (may 23): token, segment, position embeddings, input embeddings sum,
  tests, padding mask, attention mask, token type ids, input features,
  preprocessing. commits 21 to 30.
- day 4 (may 24): rest.
- day 5 (may 25): corpus reader, sentence pair sampler, nsp labels, mlm masking
  80/10/10, pre-training example, dataset, collator. commits 31 to 40.
- day 6 (may 26): dataloader factory, dynamic padding, glue base, mnli, qqp,
  qnli, sst-2, cola, sts-b, mrpc. commits 41 to 50.
- day 7 (may 27): rte, wnli, glue processor, glue collator, squad v1.1 reader,
  span extraction, feature converter, squad dataset, swag reader, swag dataset.
  commits 51 to 60.

milestone (may 27): tokenizer, input embeddings, and every dataset loader have
passing unit tests.

## phase 2 — transformer architecture (may 28 to jun 3, commits 61 to 120)

goal: scaled dot-product attention, multi-head attention, ffn, layer norm,
residual, dropout, encoder layer, bert encoder, pooler, bert model, init,
save/load, and shape tests.

- day 8 (may 28): scaled dot-product attention, mask, dropout, scores, output
  projection, multi-head, split/merge, causal mask for ltr, tests. commits 61
  to 70.
- day 9 (may 29): gelu, ffn, layer norm, residual, dropout wrapper, tests.
  commits 71 to 80.
- day 10 (may 30): encoder layer, self-attention sublayer, ffn sublayer,
  post-ln, forward, tests, gradient flow, extraction utils, config test.
  commits 81 to 90.
- day 11 (may 31): bert encoder stack, pooler, bert model, forward, hidden
  states, attention weights, init, param count, tests, integration test.
  commits 91 to 100.
- day 12 (jun 1): rest.
- day 13 (jun 2): normal init std=0.02, xavier uniform, weight tying, gradient
  clipping, save/load, checkpoint, config serialization, device transfer,
  tests. commits 101 to 110.
- day 14 (jun 3): base config test, large config test, attention shape,
  embedding shape, encoder output shape, pooler shape, batch, variable length,
  mask correctness, gradient backprop. commits 111 to 120.

milestone (jun 3): bert base and bert large forward passes produce the right
shapes. gradient flows end to end. save/load round-trips.

## phase 3 — pre-training (jun 4 to jun 10, commits 121 to 180)

goal: mlm head, nsp head, combined pre-training model, optimizer, scheduler,
trainer, pre-training script, and convergence tests.

- day 15 (jun 4): mlm head, decoder weight tying, bias, prediction, loss,
  tests, integration. commits 121 to 130.
- day 16 (jun 5): nsp head, binary classification, loss, tests, accuracy,
  combined mlm+nsp model, pre-training loss, tests. commits 131 to 140.
- day 17 (jun 6): adam with weight decay, warmup, linear decay, combined
  scheduler, gradient accumulation, tests. commits 141 to 150.
- day 18 (jun 7): rest.
- day 19 (jun 8): pre-training trainer, train step, eval step, checkpoint
  save/load, logging, loss tracking, grad norm, lr logging, tests. commits 151
  to 160.
- day 20 (jun 9): pre-training script entry, arg parser, config load, data
  load, model init, distributed, mixed precision, resume, early stop,
  integration test. commits 161 to 170.
- day 21 (jun 10): overfit test, mlm loss convergence, nsp accuracy
  convergence, gradient flow, checkpoint round-trip, mixed precision, data
  speed, smoke test, mini-run, config validation. commits 171 to 180.

milestone (jun 10): a 1-step and 10-step pre-training run completes. mlm loss
decreases on a small batch. nsp accuracy rises above chance.

## phase 4 — fine-tuning glue (jun 11 to jun 17, commits 181 to 240)

goal: classification and regression heads, fine-tuning trainer, glue script,
metrics, and per-task evaluators.

- day 22 (jun 11): classification head, dropout, loss, binary head, multiclass
  head, regression head, tests. commits 181 to 190.
- day 23 (jun 12): fine-tuning trainer, train step, eval step, early stop, best
  model, lr search, random restarts, tests. commits 191 to 200.
- day 24 (jun 13): glue script entry, arg parser, task selection, data load,
  model init, training loop, eval, prediction, lr sweep, integration test.
  commits 201 to 210.
- day 25 (jun 14): rest.
- day 26 (jun 15): accuracy, f1, precision, recall, spearman, mcc, confusion
  matrix, classification report, tests. commits 211 to 220.
- day 27 (jun 16): glue eval script, evaluator base, mnli, qqp, qnli, sst-2,
  cola, sts-b, mrpc, rte evaluators. commits 221 to 230.
- day 28 (jun 17): glue average, results table, csv export, per-task confusion,
  tests, mnli test, sst-2 test, mrpc f1 test, sts-b spearman test, integration.
  commits 231 to 240.

milestone (jun 17): each glue task has a head, an evaluator, and a metric test.
the glue script runs end to end on a tiny sample.

## phase 5 — fine-tuning qa and swag (jun 18 to jun 24, commits 241 to 300)

goal: qa span head, squad v1.1 and v2.0 scripts and eval, swag multiple choice
head and script and eval.

- day 29 (jun 18): qa span head, start and end prediction, loss, span scoring,
  best span, tests. commits 241 to 250.
- day 30 (jun 19): rest.
- day 31 (jun 20): squad script entry, arg parser, data load, model init,
  training loop, eval, prediction, answer text, n-best, integration test.
  commits 251 to 260.
- day 32 (jun 21): squad v2.0 null score, no-answer cls span, null vs non-null,
  threshold tau, v2.0 prediction, v2.0 data loader, unanswerable handling, v2.0
  script, eval, threshold optimization. commits 261 to 270.
- day 33 (jun 22): squad em, f1, normalization, answer comparison, v1.1 eval,
  v2.0 eval, best span validation, prediction format, tests. commits 271 to
  280.
- day 34 (jun 23): swag multiple choice head, score, softmax, loss, 4-choice
  input, collator, script, training loop, eval, prediction. commits 281 to
  290.
- day 35 (jun 24): swag accuracy, eval script, tests, head tests, loss test,
  integration, script integration, per-choice analysis, confusion matrix,
  results export. commits 291 to 300.

milestone (jun 24): squad v1.1, v2.0, and swag scripts run end to end on tiny
samples. em, f1, and accuracy metrics pass unit tests.

## phase 6 — ablations and analysis (jun 25 to jul 1, commits 301 to 360)

goal: pre-training task ablations, model size ablations, masking ablations,
ner feature-based ablations, training step ablations, and visualization
scripts.

- day 36 (jun 25): no nsp variant and config and script, ltr mask, ltr variant
  and config and script, ltr + bilstm, bilstm classifier, task ablation runner.
  commits 301 to 310.
- day 37 (jun 26): small, medium, medium-alt, large configs, size ablation
  runner, lm perplexity, comparison table, param count, tests. commits 311 to
  320.
- day 38 (jun 27): 100% mask, 80/0/20, 80/20/0, 0/20/80, 0/0/100, configurable
  masking class, masking runner, comparison table, tests. commits 321 to 330.
- day 39 (jun 28): rest.
- day 40 (jun 29): conll-2003 ner reader, ner label vocab, feature converter
  first sub-token, ner collator, feature extraction from layers, concat last
  4, weighted sum last 4, weighted sum all 12, bilstm ner, ner classifier.
  commits 331 to 340.
- day 41 (jun 30): ner f1 entity-level, eval script, confusion matrix, ner
  fine-tuning script, feature-based script, training loop, prediction,
  feature-based vs fine-tuning comparison, ner ablation runner, tests. commits
  341 to 350.
- day 42 (jul 1): training step ablation script, checkpoint evaluator, mnli dev
  accuracy plot, mlm vs ltr plot, attention visualization, embedding t-sne,
  glue bar chart, squad comparison table, model size plot, final results and
  readme. commits 351 to 360.

milestone (jul 1): all ablation runners execute. visualization scripts produce
plots. readme holds the final results summary.

## commit structure

- one logical change per commit. maximize the commit count.
- each commit uses the exact date and time from `timetable.md`.
- the agent may add or drop intermediate commits to fit a day, but must keep
  the phase boundaries and stay close to 360 total.
- scope taxonomy: init, config, data, model, training, scripts, eval,
  ablation, docs.
- the timetable already names each commit subject. follow it. adjust only when
  a commit must split or merge.

## standard commit format

subject: `<scope>: <imperative phrase>`
- lowercase, imperative, no period, no filler.
- scope from the taxonomy above.

body (optional, one short paragraph):
- one to three sentences. simple tense. active voice. lowercase.
- state what changed and how it maps to the paper.
- max 25 words per sentence.

canonical source: `agents.md` section "commit format".

## commit audit checks

run before every commit. fix every hit.
1. subject scope is in the allowed list.
2. subject is lowercase, imperative, no trailing period, no filler.
3. body is one paragraph, at most three sentences, each at most 25 words.
4. no contractions: 'll, 're, 's, n't, it's.
5. no perfect tenses: has been, have been, had been.
6. no banned modals: should, would, may, might, could.
7. no "-ing" clauses after a comma: , making, , allowing, , enabling.
8. no semicolons.
9. no latin abbreviations: e.g., i.e., etc.
10. no filler: simply, easily, seamlessly, robust, leverage, utilize,
    comprehensive.
11. no mention of dataset or downloader scripts.
12. commit date matches the timetable day.

## verification gates

the agent must not start the next phase until the current gate passes.
- gate 1 (may 27): tokenizer and data pipeline tests pass.
- gate 2 (jun 3): bert model forward and shape tests pass.
- gate 3 (jun 10): pre-training smoke test and loss tests pass.
- gate 4 (jun 17): glue heads, metrics, and evaluators tests pass.
- gate 5 (jun 24): qa, swag heads and eval tests pass.
- gate 6 (jul 1): ablation runners and analysis scripts run.

## risk notes

- the paper omits some implementation details (for example, exact pooler init,
  tokenizer edge cases). when the paper is silent, keep the simplest choice
  and note it in the commit body. do not invent paper values.
- squad v2.0 and the masking ablations depend on v1.1 and the base masking
  code. build those first.
- the feature-based ner ablation depends on a trained checkpoint. if no real
  checkpoint exists, run the ablation runner on a randomly initialized model
  and mark the results as a smoke test.
