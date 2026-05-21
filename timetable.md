# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — Implementation Timetable

**Paper**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.
**Start**: May 21, 2026 | **End**: July 1, 2026
**Working days**: 36 | **Rest days**: 6 | **Total commits**: 360

---

## PHASE 1: Project Scaffolding & Data Pipeline (May 21 – May 27)

### Day 1 — Thursday May 21

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 1    | 00:01:02 | `init: initialize project structure with directories`              |
| 2    | 00:06:39 | `init: add README.md with project overview`                         |
| 3    | 00:48:59 | `init: add requirements.txt with PyTorch dependencies`              |
| 4    | 01:10:00 | `init: add setup.py for package installation`                      |
| 5    | 01:22:00 | `init: add .gitignore for Python project`                          |
| 6    | 02:31:44 | `config: add BERT model configuration dataclass`                    |
| 7    | 03:30:19 | `config: add BERT_BASE configuration preset (L=12 H=768 A=12)`     |
| 8    | 21:11:25 | `config: add BERT_LARGE configuration preset (L=24 H=1024 A=16)`   |
| 9    | 23:03:00 | `config: add pre-training configuration dataclass`                  |
| 10   | 23:43:36 | `config: add fine-tuning configuration dataclass`                   |

### Day 2 — Friday May 22

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 11   | 01:20:17 | `data: add special token constants ([CLS] [SEP] <tool_call> [PAD] [UNK])`    |
| 12   | 01:51:19 | `data: add vocabulary class for token-to-id mapping`                |
| 13   | 01:52:12 | `data: add WordPiece tokenizer base class`                          |
| 14   | 02:43:57 | `data: add wordpiece greedy longest-match-first algorithm`          |
| 15   | 02:47:12 | `data: add subword segmentation with ## prefix handling`            |
| 16   | 02:54:37 | `data: add tokenizer encode method for single text`                 |
| 17   | 02:58:13 | `data: add tokenizer encode_pair method for sentence pairs`         |
| 18   | 21:31:07 | `data: add tokenizer decode method for id-to-text`                 |
| 19   | 22:02:34 | `data: add vocabulary builder from training corpus`                 |
| 20   | 22:12:18 | `data: add tokenizer unit tests`                                    |

### Day 3 — Saturday May 23

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 21   | 00:03:36 | `model: add token embedding layer class`                            |
| 22   | 01:09:11 | `model: add segment embedding layer class`                          |
| 23   | 02:54:49 | `model: add position embedding layer class`                         |
| 24   | 21:06:46 | `model: add BERT input embeddings (sum of token+segment+position)` |
| 25   | 22:41:52 | `model: add input embeddings unit tests`                           |
| 26   | 22:51:31 | `data: add padding mask creation utility`                          |
| 27   | 22:56:36 | `data: add attention mask creation utility`                        |
| 28   | 23:01:13 | `data: add token type id creation utility`                          |
| 29   | 23:02:05 | `data: add input features dataclass`                                |
| 30   | 23:11:34 | `data: add input preprocessing pipeline`                           |

### Day 4 — Sunday May 24 — REST

### Day 5 — Monday May 25

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 31   | 00:31:21 | `data: add document corpus reader`                                  |
| 32   | 01:33:05 | `data: add sentence pair sampler for NSP`                           |
| 33   | 02:06:36 | `data: add NSP label assignment (IsNext/NotNext)`                   |
| 34   | 02:28:57 | `data: add MLM masking strategy (80% replace with <tool_call>)`             |
| 35   | 02:29:12 | `data: add MLM random token replacement (10%)`                      |
| 36   | 02:35:55 | `data: add MLM keep unchanged (10%)`                                |
| 37   | 21:59:56 | `data: add combined MLM masking procedure`                         |
| 38   | 22:15:29 | `data: add pre-training example dataclass`                          |
| 39   | 22:20:10 | `data: add pre-training dataset class`                              |
| 40   | 23:40:26 | `data: add pre-training data collator`                              |

### Day 6 — Tuesday May 26

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 41   | 01:37:07 | `data: add PyTorch DataLoader factory for pre-training`             |
| 42   | 01:41:37 | `data: add batch creation with dynamic padding`                    |
| 43   | 01:57:25 | `data: add GLUE task base class`                                    |
| 44   | 02:58:32 | `data: add MNLI dataset loader`                                     |
| 45   | 22:17:40 | `data: add QQP dataset loader`                                      |
| 46   | 22:23:23 | `data: add QNLI dataset loader`                                     |
| 47   | 22:46:09 | `data: add SST-2 dataset loader`                                    |
| 48   | 22:55:43 | `data: add CoLA dataset loader`                                    |
| 49   | 23:06:07 | `data: add STS-B dataset loader`                                    |
| 50   | 23:59:43 | `data: add MRPC dataset loader`                                     |

### Day 7 — Wednesday May 27

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 51   | 00:25:56 | `data: add RTE dataset loader`                                      |
| 52   | 01:20:43 | `data: add WNLI dataset loader`                                     |
| 53   | 01:59:51 | `data: add GLUE data processor for train/dev/test splits`           |
| 54   | 02:17:33 | `data: add GLUE data collator`                                      |
| 55   | 02:27:42 | `data: add SQuAD v1.1 data reader`                                  |
| 56   | 03:43:52 | `data: add SQuAD answer span extraction`                            |
| 57   | 21:34:27 | `data: add SQuAD feature converter`                                 |
| 58   | 22:35:08 | `data: add SQuAD dataset class`                                     |
| 59   | 23:22:39 | `data: add SWAG data reader`                                        |
| 60   | 23:35:13 | `data: add SWAG dataset class`                                      |

---

## PHASE 2: Transformer Architecture (May 28 – June 3)

### Day 8 — Thursday May 28

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 61   | 00:08:50 | `model: add scaled dot-product attention function`                 |
| 62   | 01:14:39 | `model: add attention mask application`                            |
| 63   | 01:42:52 | `model: add attention dropout`                                      |
| 64   | 02:24:59 | `model: add attention scores computation`                          |
| 65   | 02:52:08 | `model: add attention output projection`                           |
| 66   | 03:22:56 | `model: add multi-head attention class`                            |
| 67   | 21:03:41 | `model: add multi-head split and merge`                             |
| 68   | 21:23:01 | `model: add causal mask for LTR ablation`                           |
| 69   | 22:24:06 | `model: add attention unit tests`                                  |
| 70   | 22:44:31 | `model: add attention numerical stability test`                     |

### Day 9 — Friday May 29

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 71   | 00:10:50 | `model: add GELU activation function`                              |
| 72   | 01:04:23 | `model: add position-wise feed-forward network`                    |
| 73   | 01:06:27 | `model: add layer normalization class`                             |
| 74   | 01:10:46 | `model: add residual connection wrapper`                           |
| 75   | 01:33:35 | `model: add dropout layer wrapper`                                  |
| 76   | 01:42:17 | `model: add FFN unit tests`                                         |
| 77   | 02:24:10 | `model: add layer norm unit tests`                                  |
| 78   | 02:44:52 | `model: add GELU unit tests`                                        |
| 79   | 02:46:24 | `model: add residual connection unit tests`                         |
| 80   | 23:33:05 | `model: add numerical stability tests for FFN`                      |

### Day 10 — Saturday May 30

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 81   | 00:25:18 | `model: add transformer encoder layer class`                       |
| 82   | 00:28:35 | `model: add self-attention sublayer`                               |
| 83   | 00:41:07 | `model: add feed-forward sublayer`                                  |
| 84   | 01:07:35 | `model: add layer norm placement (post-LN)`                        |
| 85   | 02:33:57 | `model: add encoder layer forward pass`                             |
| 86   | 03:49:12 | `model: add encoder layer unit tests`                               |
| 87   | 22:10:20 | `model: add gradient flow test through encoder layer`               |
| 88   | 23:10:41 | `model: add attention weight extraction`                            |
| 89   | 23:17:46 | `model: add hidden state extraction utility`                        |
| 90   | 23:50:22 | `model: add encoder layer config test`                              |

### Day 11 — Sunday May 31

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 91   | 01:51:53 | `model: add BERT encoder (stack of transformer layers)`            |
| 92   | 01:54:24 | `model: add BERT pooler (tanh on [CLS])`                            |
| 93   | 02:03:33 | `model: add BERT model class`                                       |
| 94   | 02:07:58 | `model: add BERT forward pass`                                      |
| 95   | 02:21:02 | `model: add BERT hidden states output`                              |
| 96   | 02:25:55 | `model: add BERT attention weights output`                          |
| 97   | 02:25:57 | `model: add BERT parameter initialization`                         |
| 98   | 02:30:01 | `model: add BERT parameter count utility`                          |
| 99   | 21:27:44 | `model: add BERT model unit tests`                                  |
| 100  | 22:44:36 | `model: add BERT integration test`                                  |

### Day 12 — Monday June 1 — REST

### Day 13 — Tuesday June 2

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 101  | 00:27:24 | `model: add weight initialization (normal std=0.02)`                |
| 102  | 00:53:49 | `model: add Xavier uniform initialization option`                  |
| 103  | 01:11:52 | `model: add embedding weight tying option`                         |
| 104  | 01:22:47 | `model: add gradient clipping utility`                             |
| 105  | 02:24:17 | `model: add model save/load utilities`                             |
| 106  | 02:54:05 | `model: add checkpoint serialization`                             |
| 107  | 21:50:50 | `model: add model config serialization`                           |
| 108  | 22:00:52 | `model: add model device transfer`                                  |
| 109  | 22:11:55 | `model: add parameter counting test`                               |
| 110  | 22:22:28 | `model: add model save/load round-trip test`                        |

### Day 14 — Wednesday June 3

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 111  | 00:01:20 | `model: add BERT_BASE config validation test`                      |
| 112  | 00:12:52 | `model: add BERT_LARGE config validation test`                     |
| 113  | 00:28:25 | `model: add attention shape verification test`                     |
| 114  | 00:33:27 | `model: add embedding shape verification test`                     |
| 115  | 01:24:03 | `model: add encoder output shape test`                             |
| 116  | 02:17:25 | `model: add pooler output shape test`                               |
| 117  | 02:46:35 | `model: add batch processing test`                                  |
| 118  | 02:49:57 | `model: add variable sequence length test`                         |
| 119  | 02:52:32 | `model: add attention mask correctness test`                       |
| 120  | 23:56:32 | `model: add gradient backpropagation test`                         |

---

## PHASE 3: Pre-training (June 4 – June 10)

### Day 15 — Thursday June 4

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 121  | 00:28:31 | `model: add MLM head (linear + gelu + layernorm)`                  |
| 122  | 00:44:47 | `model: add MLM decoder (weight tying with embeddings)`            |
| 123  | 01:16:29 | `model: add MLM bias parameter`                                    |
| 124  | 01:17:29 | `model: add MLM prediction computation`                            |
| 125  | 02:08:17 | `model: add MLM loss computation (cross-entropy)`                  |
| 126  | 02:11:29 | `model: add MLM head unit tests`                                    |
| 127  | 02:23:05 | `model: add MLM loss test`                                          |
| 128  | 02:26:27 | `model: add MLM prediction shape test`                             |
| 129  | 02:37:31 | `model: add MLM weight tying test`                                  |
| 130  | 22:38:38 | `model: add MLM integration test with BERT`                        |

### Day 16 — Friday June 5

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 131  | 00:59:21 | `model: add NSP head (pooler output + linear)`                     |
| 132  | 01:03:48 | `model: add NSP binary classification`                            |
| 133  | 01:31:00 | `model: add NSP loss computation (cross-entropy)`                  |
| 134  | 01:54:31 | `model: add NSP head unit tests`                                    |
| 135  | 02:59:27 | `model: add NSP loss test`                                          |
| 136  | 21:02:10 | `model: add NSP prediction test`                                    |
| 137  | 21:59:27 | `model: add NSP accuracy metric`                                    |
| 138  | 22:03:19 | `model: add combined MLM+NSP pre-training model`                   |
| 139  | 22:22:16 | `model: add pre-training loss computation`                         |
| 140  | 22:51:30 | `model: add pre-training model unit tests`                          |

### Day 17 — Saturday June 6

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 141  | 00:56:24 | `training: add Adam optimizer wrapper with weight decay`            |
| 142  | 01:22:32 | `training: add learning rate warmup schedule`                      |
| 143  | 01:28:03 | `training: add linear decay schedule`                              |
| 144  | 01:36:26 | `training: add combined warmup + linear decay scheduler`           |
| 145  | 02:06:05 | `training: add gradient accumulation utility`                      |
| 146  | 02:08:33 | `training: add optimizer unit tests`                               |
| 147  | 02:58:09 | `training: add scheduler unit tests`                               |
| 148  | 21:02:58 | `training: add warmup schedule test`                               |
| 149  | 22:03:59 | `training: add linear decay test`                                  |
| 150  | 23:49:14 | `training: add gradient clipping test`                             |

### Day 18 — Sunday June 7 — REST

### Day 19 — Monday June 8

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 151  | 00:47:50 | `training: add pre-training trainer class`                         |
| 152  | 00:52:06 | `training: add pre-training training step`                          |
| 153  | 00:57:15 | `training: add pre-training evaluation step`                       |
| 154  | 01:13:42 | `training: add pre-training checkpoint saving`                     |
| 155  | 01:46:01 | `training: add pre-training checkpoint loading`                    |
| 156  | 21:27:31 | `training: add pre-training logging`                                |
| 157  | 22:39:25 | `training: add pre-training loss tracking`                         |
| 158  | 22:43:58 | `training: add pre-training gradient norm logging`                  |
| 159  | 23:05:54 | `training: add pre-training learning rate logging`                 |
| 160  | 23:30:13 | `training: add pre-training trainer unit tests`                    |

### Day 20 — Tuesday June 9

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 161  | 00:24:44 | `scripts: add pre-training script entry point`                     |
| 162  | 00:47:33 | `scripts: add pre-training argument parser`                        |
| 163  | 01:29:57 | `scripts: add pre-training config loading`                          |
| 164  | 01:31:08 | `scripts: add pre-training data loading`                           |
| 165  | 01:55:21 | `scripts: add pre-training model initialization`                    |
| 166  | 03:32:23 | `scripts: add pre-training distributed option`                     |
| 167  | 21:08:18 | `scripts: add pre-training mixed precision option`                 |
| 168  | 22:25:27 | `scripts: add pre-training resume from checkpoint`                  |
| 169  | 22:52:07 | `scripts: add pre-training early stopping`                         |
| 170  | 23:31:47 | `scripts: add pre-training script integration test`                |

### Day 21 — Wednesday June 10

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 171  | 00:31:33 | `training: add pre-training overfit test on small batch`            |
| 172  | 01:11:26 | `training: add MLM loss convergence test`                          |
| 173  | 01:59:03 | `training: add NSP accuracy convergence test`                     |
| 174  | 02:14:02 | `training: add gradient flow test through full model`              |
| 175  | 02:46:14 | `training: add checkpoint round-trip test`                        |
| 176  | 22:43:24 | `training: add mixed precision training test`                      |
| 177  | 22:52:45 | `training: add data loading speed test`                           |
| 178  | 23:26:40 | `training: add pre-training smoke test (1 step)`                    |
| 179  | 23:31:48 | `training: add pre-training mini-run test (10 steps)`              |
| 180  | 23:43:54 | `training: add pre-training config validation test`                |

---

## PHASE 4: Fine-tuning — GLUE (June 11 – June 17)

### Day 22 — Thursday June 11

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 181  | 01:03:42 | `model: add classification head for GLUE`                          |
| 182  | 01:58:07 | `model: add classification head dropout`                           |
| 183  | 02:52:57 | `model: add classification loss computation`                       |
| 184  | 02:57:11 | `model: add binary classification head`                            |
| 185  | 21:38:34 | `model: add multiclass classification head`                        |
| 186  | 22:29:23 | `model: add regression head for STS-B`                             |
| 187  | 22:35:21 | `model: add classification head unit tests`                        |
| 188  | 22:50:43 | `model: add classification loss test`                              |
| 189  | 23:10:08 | `model: add regression loss test`                                  |
| 190  | 23:57:11 | `model: add classification head integration test`                  |

### Day 23 — Friday June 12

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 191  | 00:17:35 | `training: add fine-tuning trainer class`                          |
| 192  | 00:20:42 | `training: add fine-tuning training step`                          |
| 193  | 00:37:21 | `training: add fine-tuning evaluation step`                        |
| 194  | 01:25:46 | `training: add fine-tuning early stopping`                         |
| 195  | 01:27:46 | `training: add fine-tuning best model selection`                   |
| 196  | 01:40:14 | `training: add fine-tuning learning rate search`                   |
| 197  | 02:13:53 | `training: add fine-tuning random restarts`                        |
| 198  | 02:18:37 | `training: add fine-tuning trainer unit tests`                     |
| 199  | 03:25:47 | `training: add fine-tuning evaluation test`                        |
| 200  | 22:56:45 | `training: add fine-tuning checkpoint test`                        |

### Day 24 — Saturday June 13

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 201  | 00:04:29 | `scripts: add GLUE fine-tuning script entry point`                  |
| 202  | 00:10:20 | `scripts: add GLUE task argument parser`                           |
| 203  | 00:33:02 | `scripts: add GLUE task selection`                                  |
| 204  | 00:39:35 | `scripts: add GLUE data loading`                                    |
| 205  | 00:42:45 | `scripts: add GLUE model initialization`                           |
| 206  | 03:05:28 | `scripts: add GLUE training loop`                                  |
| 207  | 22:24:01 | `scripts: add GLUE evaluation`                                      |
| 208  | 22:36:02 | `scripts: add GLUE prediction output`                               |
| 209  | 22:59:54 | `scripts: add GLUE learning rate sweep`                            |
| 210  | 23:49:50 | `scripts: add GLUE script integration test`                        |

### Day 25 — Sunday June 14 — REST

### Day 26 — Monday June 15

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 211  | 00:13:20 | `eval: add accuracy metric`                                         |
| 212  | 01:05:27 | `eval: add F1 score metric`                                         |
| 213  | 01:40:51 | `eval: add precision metric`                                       |
| 214  | 02:27:19 | `eval: add recall metric`                                           |
| 215  | 21:56:25 | `eval: add Spearman correlation metric`                            |
| 216  | 22:34:03 | `eval: add Matthews correlation coefficient`                       |
| 217  | 23:11:13 | `eval: add confusion matrix`                                        |
| 218  | 23:37:00 | `eval: add classification report`                                  |
| 219  | 23:38:00 | `eval: add metric unit tests`                                       |
| 220  | 23:40:42 | `eval: add F1 score test`                                           |

### Day 27 — Tuesday June 16

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 221  | 00:15:13 | `eval: add GLUE evaluation script`                                  |
| 222  | 00:31:44 | `eval: add GLUE task evaluator base class`                         |
| 223  | 00:34:57 | `eval: add MNLI evaluator`                                          |
| 224  | 02:02:49 | `eval: add QQP evaluator`                                           |
| 225  | 02:38:18 | `eval: add QNLI evaluator`                                          |
| 226  | 03:45:24 | `eval: add SST-2 evaluator`                                         |
| 227  | 21:25:31 | `eval: add CoLA evaluator`                                         |
| 228  | 22:27:24 | `eval: add STS-B evaluator`                                         |
| 229  | 22:50:20 | `eval: add MRPC evaluator`                                          |
| 230  | 23:19:07 | `eval: add RTE evaluator`                                           |

### Day 28 — Wednesday June 17

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 231  | 01:19:35 | `eval: add GLUE average score computation`                         |
| 232  | 01:21:57 | `eval: add GLUE results table generation`                          |
| 233  | 02:50:16 | `eval: add GLUE results CSV export`                                |
| 234  | 03:28:58 | `eval: add GLUE per-task confusion matrix`                         |
| 235  | 03:40:23 | `eval: add GLUE evaluation unit tests`                             |
| 236  | 03:51:11 | `eval: add MNLI evaluation test`                                    |
| 237  | 21:14:44 | `eval: add SST-2 evaluation test`                                  |
| 238  | 21:41:34 | `eval: add MRPC F1 test`                                            |
| 239  | 23:07:42 | `eval: add STS-B Spearman test`                                    |
| 240  | 23:27:57 | `eval: add GLUE integration test`                                  |

---

## PHASE 5: Fine-tuning — QA & SWAG (June 18 – June 24)

### Day 29 — Thursday June 18

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 241  | 00:06:18 | `model: add QA span prediction head`                               |
| 242  | 00:33:04 | `model: add start position prediction`                            |
| 243  | 00:58:23 | `model: add end position prediction`                              |
| 244  | 02:13:04 | `model: add QA start/end loss computation`                         |
| 245  | 02:24:35 | `model: add QA span scoring (S dot T_i + E dot T_j)`               |
| 246  | 02:25:16 | `model: add QA best span selection`                                |
| 247  | 02:29:46 | `model: add QA head unit tests`                                     |
| 248  | 22:11:45 | `model: add QA loss test`                                           |
| 249  | 22:31:48 | `model: add QA span selection test`                                 |
| 250  | 23:02:44 | `model: add QA head integration test`                              |

### Day 30 — Friday June 19 — REST

### Day 31 — Saturday June 20

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 251  | 00:28:50 | `scripts: add SQuAD fine-tuning script entry point`                 |
| 252  | 00:32:36 | `scripts: add SQuAD argument parser`                              |
| 253  | 00:54:15 | `scripts: add SQuAD data loading`                                  |
| 254  | 01:01:23 | `scripts: add SQuAD model initialization`                          |
| 255  | 01:27:28 | `scripts: add SQuAD training loop`                                 |
| 256  | 02:22:57 | `scripts: add SQuAD evaluation`                                     |
| 257  | 03:00:22 | `scripts: add SQuAD prediction output`                              |
| 258  | 22:25:27 | `scripts: add SQuAD answer text extraction`                        |
| 259  | 22:55:43 | `scripts: add SQuAD n-best prediction filtering`                    |
| 260  | 23:04:35 | `scripts: add SQuAD script integration test`                       |

### Day 32 — Sunday June 21

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 261  | 00:29:50 | `model: add SQuAD v2.0 null score computation`                      |
| 262  | 00:54:52 | `model: add no-answer prediction (CLS span)`                       |
| 263  | 01:34:06 | `model: add null vs non-null score comparison`                     |
| 264  | 02:26:47 | `model: add threshold tau for null prediction`                     |
| 265  | 03:32:39 | `model: add SQuAD v2.0 prediction logic`                           |
| 266  | 03:35:49 | `data: add SQuAD v2.0 data loader`                                  |
| 267  | 21:00:07 | `data: add SQuAD v2.0 unanswerable handling`                       |
| 268  | 21:04:09 | `scripts: add SQuAD v2.0 fine-tuning script`                       |
| 269  | 22:17:43 | `scripts: add SQuAD v2.0 evaluation`                              |
| 270  | 23:46:13 | `scripts: add SQuAD v2.0 threshold optimization`                   |

### Day 33 — Monday June 22

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 271  | 00:42:51 | `eval: add SQuAD EM (Exact Match) metric`                          |
| 272  | 01:18:18 | `eval: add SQuAD F1 metric`                                         |
| 273  | 01:32:07 | `eval: add SQuAD normalization function`                           |
| 274  | 01:59:02 | `eval: add SQuAD answer comparison`                                |
| 275  | 02:11:18 | `eval: add SQuAD v1.1 evaluation script`                           |
| 276  | 02:50:33 | `eval: add SQuAD v2.0 evaluation with no-answer`                   |
| 277  | 22:23:18 | `eval: add SQuAD best span validation`                             |
| 278  | 23:23:08 | `eval: add SQuAD prediction format`                                |
| 279  | 23:41:15 | `eval: add SQuAD evaluation unit tests`                            |
| 280  | 23:45:58 | `eval: add SQuAD F1 computation test`                             |

### Day 34 — Tuesday June 23

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 281  | 00:26:35 | `model: add multiple choice head for SWAG`                          |
| 282  | 01:09:09 | `model: add SWAG score computation`                                |
| 283  | 01:37:21 | `model: add SWAG softmax over 4 choices`                           |
| 284  | 02:22:40 | `model: add SWAG loss computation`                                |
| 285  | 02:31:53 | `data: add SWAG 4-choice input construction`                       |
| 286  | 21:37:18 | `data: add SWAG data collator`                                     |
| 287  | 22:02:59 | `scripts: add SWAG fine-tuning script`                             |
| 288  | 22:24:15 | `scripts: add SWAG training loop`                                  |
| 289  | 23:34:13 | `scripts: add SWAG evaluation`                                     |
| 290  | 23:45:57 | `scripts: add SWAG prediction output`                              |

### Day 35 — Wednesday June 24

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 291  | 00:07:34 | `eval: add SWAG accuracy metric`                                    |
| 292  | 00:14:04 | `eval: add SWAG evaluation script`                                 |
| 293  | 01:17:55 | `eval: add SWAG evaluation unit tests`                             |
| 294  | 01:55:35 | `model: add SWAG head unit tests`                                   |
| 295  | 02:46:31 | `model: add SWAG loss test`                                         |
| 296  | 21:21:06 | `model: add SWAG integration test`                                 |
| 297  | 22:17:27 | `scripts: add SWAG script integration test`                        |
| 298  | 23:00:34 | `eval: add SWAG per-choice analysis`                                |
| 299  | 23:09:12 | `eval: add SWAG confusion matrix`                                  |
| 300  | 23:39:38 | `eval: add SWAG results export`                                    |

---

## PHASE 6: Ablations, Experiments & Analysis (June 25 – July 1)

### Day 36 — Thursday June 25

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 301  | 00:07:36 | `ablation: add No NSP model variant`                               |
| 302  | 01:00:07 | `ablation: add No NSP training config`                             |
| 303  | 01:38:58 | `ablation: add No NSP pre-training script`                         |
| 304  | 02:11:34 | `ablation: add LTR (left-to-right) attention mask`                 |
| 305  | 03:36:52 | `ablation: add LTR model variant`                                  |
| 306  | 03:58:50 | `ablation: add LTR training config`                                |
| 307  | 21:31:41 | `ablation: add LTR pre-training script`                            |
| 308  | 22:45:09 | `ablation: add LTR + BiLSTM fine-tuning variant`                   |
| 309  | 23:19:56 | `ablation: add BiLSTM classifier head`                             |
| 310  | 23:55:07 | `ablation: add pre-training task ablation runner`                  |

### Day 37 — Friday June 26

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 311  | 00:19:29 | `ablation: add small model config (L=3 H=768 A=12)`                |
| 312  | 00:25:24 | `ablation: add medium model config (L=6 H=768 A=3)`                |
| 313  | 01:21:20 | `ablation: add medium model config (L=6 H=768 A=12)`                |
| 314  | 02:03:57 | `ablation: add large model config (L=12 H=1024 A=16)`               |
| 315  | 02:24:31 | `ablation: add model size ablation runner`                         |
| 316  | 21:25:38 | `ablation: add LM perplexity computation`                          |
| 317  | 21:45:07 | `ablation: add model size comparison table`                        |
| 318  | 22:14:02 | `ablation: add model parameter count comparison`                   |
| 319  | 23:05:40 | `ablation: add model size ablation tests`                          |
| 320  | 23:35:26 | `ablation: add perplexity computation test`                       |

### Day 38 — Saturday June 27

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 321  | 00:01:18 | `ablation: add 100% mask strategy variant`                         |
| 322  | 00:13:49 | `ablation: add 80/0/20 mask strategy variant`                     |
| 323  | 00:28:15 | `ablation: add 80/20/0 mask strategy variant`                     |
| 324  | 00:30:34 | `ablation: add 0/20/80 mask strategy variant`                     |
| 325  | 01:28:43 | `ablation: add 0/0/100 mask strategy variant`                     |
| 326  | 01:31:17 | `ablation: add configurable masking strategy class`                |
| 327  | 03:54:48 | `ablation: add masking ablation runner`                            |
| 328  | 22:11:24 | `ablation: add masking strategy comparison table`                 |
| 329  | 22:50:47 | `ablation: add masking ablation tests`                            |
| 330  | 23:06:24 | `ablation: add masking strategy validation test`                   |

### Day 39 — Sunday June 28 — REST

### Day 40 — Monday June 29

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 331  | 01:02:20 | `data: add CoNLL-2003 NER data reader`                             |
| 332  | 01:04:59 | `data: add NER label vocabulary`                                   |
| 333  | 01:33:38 | `data: add NER feature converter (first sub-token)`                |
| 334  | 01:34:56 | `data: add NER data collator`                                      |
| 335  | 01:55:30 | `ablation: add feature extraction from BERT layers`                |
| 336  | 02:25:11 | `ablation: add concat last 4 layers feature`                       |
| 337  | 02:28:00 | `ablation: add weighted sum last 4 layers feature`                 |
| 338  | 02:32:35 | `ablation: add weighted sum all 12 layers feature`                 |
| 339  | 23:30:45 | `ablation: add BiLSTM NER classifier`                              |
| 340  | 23:51:21 | `ablation: add NER fine-tuning classifier`                         |

### Day 41 — Tuesday June 30

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 341  | 00:18:13 | `eval: add NER F1 metric (entity-level)`                           |
| 342  | 00:39:58 | `eval: add NER evaluation script`                                  |
| 343  | 00:46:18 | `eval: add NER confusion matrix`                                   |
| 344  | 01:35:19 | `scripts: add NER fine-tuning script`                              |
| 345  | 01:42:20 | `scripts: add NER feature-based script`                            |
| 346  | 01:43:55 | `scripts: add NER training loop`                                   |
| 347  | 02:52:38 | `scripts: add NER prediction output`                               |
| 348  | 22:50:27 | `ablation: add feature-based vs fine-tuning comparison`            |
| 349  | 22:56:46 | `ablation: add NER ablation runner`                                |
| 350  | 23:55:40 | `eval: add NER evaluation unit tests`                             |

### Day 42 — Wednesday July 1

| #    | Time     | Commit                                                              |
| ---- | -------- | ------------------------------------------------------------------- |
| 351  | 00:56:27 | `scripts: add training step ablation script`                       |
| 352  | 01:04:17 | `scripts: add training step checkpoint evaluator`                  |
| 353  | 02:04:33 | `scripts: add MNLI dev accuracy vs steps plot`                     |
| 354  | 03:07:27 | `scripts: add MLM vs LTR convergence plot`                         |
| 355  | 21:03:22 | `scripts: add attention weight visualization`                      |
| 356  | 21:21:13 | `scripts: add embedding visualization (t-SNE)`                     |
| 357  | 22:09:28 | `scripts: add GLUE results bar chart`                              |
| 358  | 22:12:12 | `scripts: add SQuAD results comparison table`                     |
| 359  | 23:00:14 | `scripts: add model size ablation plot`                            |
| 360  | 23:15:23 | `docs: add final results summary and README update`                |

---

## Summary

| Phase                                              | Dates              | Working | Rest | Commits |
| -------------------------------------------------- | ------------------ | ------- | ---- | ------- |
| 1. Project Scaffolding & Data Pipeline             | May 21 – May 27    | 6       | 1    | 60      |
| 2. Transformer Architecture                        | May 28 – Jun 3     | 6       | 1    | 60      |
| 3. Pre-training                                     | Jun 4 – Jun 10     | 6       | 1    | 60      |
| 4. Fine-tuning — GLUE                               | Jun 11 – Jun 17    | 6       | 1    | 60      |
| 5. Fine-tuning — QA & SWAG                          | Jun 18 – Jun 24    | 6       | 1    | 60      |
| 6. Ablations, Experiments & Analysis               | Jun 25 – Jul 1     | 6       | 1    | 60      |
| **Total**                                          | **42 days**        | **36**  | **6** | **360** |
