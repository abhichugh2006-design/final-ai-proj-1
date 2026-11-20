# Legal Document LoRA Tagger

[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ax9x3mxmwc/badge.svg)](https://ask.thili.ai/b/ax9x3mxmwc)

Clerks in government offices read the same kinds of contract clauses all day: who pays when, how a contract ends,
who covers a lawsuit. This tool reads a contract, splits it into clauses, and labels each one, so a clerk can jump
straight to the clauses they need to check.

It works by teaching GPT-2 (124M) one new skill — "read a clause, answer with a tag" — through a **LoRA adapter
written from scratch** (no `peft` library). Fewer than 1% of the model's weights are trained; the adapter is
then merged back in, so the final model runs at plain GPT-2 speed.

```
$ python -m app.cli tag --file examples/sample_contract.txt
[CONFIDENTIALITY] (confidence %)
The Vendor shall not disclose any citizen records or internal files of the Records Office ...

[PAYMENT] (confidence %)
The Records Office shall pay each approved invoice within forty-five days of receiving it.
...
```
(Real output from the trained model is in the Results section below and in `results/tagged_sample.txt`.)

The six tags: `confidentiality`, `termination`, `indemnification`, `payment`, `governing_law`, `limitation_of_liability`.

## Quick start

Needs Python 3.12 and about 1 GB of free disk space (GPT-2 is downloaded from Hugging Face the first time).

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                      # 15 tests, about 10 s, no download needed
python -m app.cli dataset                                   # see the data and the train/test split
python -m app.cli train                                     # train + merge + evaluate, a few minutes on a laptop CPU
python -m app.cli tag --file examples/sample_contract.txt   # tag a document with the trained model
python -m app.cli evaluate                                  # score the saved model against the baselines
```

`train` saves the merged model to `models/lora-tagger/` (ignored by git, about 500 MB) and writes
`results/metrics.json`, `results/RESULTS.md` and `results/tagged_sample.txt`. Add `--update-readme README.md`
to also refresh the Results section below. Everything is seeded (`--seed 0`), so reruns give the same split
and the same training order.

You can also tag any text directly: `python -m app.cli tag --text "Either party may end this agreement with 30 days' notice."`
or get JSON with `--json`.

## How it works

```
"Clause: The Purchaser shall pay ... \nTag:"  ──►  GPT-2 with LoRA on every attention block  ──►  scores for the next token
                                                                                                   │
                                          compare ONLY the 6 tag tokens, pick the highest ◄────────┘
                                                                  │
                                                     <|tag:payment|>  (confidence = softmax over the 6)
```

| Step | Code (`app/core.py`) | What happens |
|---|---|---|
| 1. Data | `load_clauses`, `train_test_split`, `format_example` | 72 labelled clauses (12 per tag) in `corpus/legal_clauses.jsonl`. Stratified 75/25 split, so every tag has 9 training and 3 test clauses. Each becomes `Clause: <text>\nTag:<|tag:x|><|endoftext|>`. |
| 2. Tag tokens | `add_tag_tokens` | Adds one special token per tag and resizes GPT-2's embeddings. New rows start as the average of a few related words (e.g. *pay invoice fee*) instead of random noise. |
| 3. LoRA | `LoRALinear`, `inject_lora` | Each target layer becomes `y = Wx + (α/r)·B(Ax)`. W is frozen; A (r×in) and B (out×r) are trained. B starts at zero, so before training the model is unchanged (a test checks this). Handles GPT-2's `Conv1D`, whose weight is stored transposed. |
| 4. Fine-tune | `encode_batch`, `train` | Loss is computed only on the tag token and end token; the clause text is masked with `-100`, so the model learns to classify, not to copy text. AdamW on the LoRA weights only. |
| 5. Merge + tag | `merge_lora`, `predict`, `split_clauses`, `tag_document` | Folds `B·A·α/r` into W and removes the adapter (a test checks the answers are identical before and after). Prediction compares only the 6 tag tokens, so the output is always a valid tag. Documents are split on blank lines and on numbering such as `1.` or `(a)`. |
| 6. Evaluate | `classification_report`, `efficiency_report`, `baselines` | Accuracy, per-tag precision/recall/F1, macro-F1, a confusion matrix and the list of mistakes, plus the number of trainable parameters and adapter size compared with full fine-tuning. |

## Data

- `corpus/legal_clauses.jsonl` — 72 clauses I wrote in the style of Indian public-sector contracts, tenders,
  leases and grant letters (12 per tag). Used for the train/test split.
- `corpus/challenge_clauses.jsonl` — 18 extra clauses (3 per tag) written on purpose **without** the obvious
  keywords (*"bring this arrangement to a close"* instead of *"terminate"*). Never used for training; it checks
  whether the model learned meaning or only keywords.
- `examples/sample_contract.txt` — a short 6-clause services agreement for trying the `tag` command.

Each line is `{"text": "...", "label": "<one of the six tags>"}`. To use your own data, pass `--corpus your.jsonl`.

## Results

These numbers are written by `python -m app.cli train --update-readme README.md`. The `train-and-report` GitHub
Action runs that command on GitHub's machine whenever the code or data changes, then commits the output here and
in `results/`.

<!-- RESULTS:START -->
*Results are being generated by the `train-and-report` GitHub Action — open the Actions tab of this repository.
It takes about 5–10 minutes after a push, then this section and `results/` are filled in automatically.*

What is already known without training (`keyword_tag` baseline, computed by `python -m app.cli evaluate` / `tests`):

| Method | Test accuracy | Test macro-F1 | Challenge accuracy | Challenge macro-F1 |
|---|---|---|---|---|
| Majority class (always the same tag) | 16.7% | 0.048 | 16.7% | 0.048 |
| Keyword rules (`keyword_tag`) | 94.4% | 0.943 | 27.8% | 0.222 |
<!-- RESULTS:END -->

### Baselines: what the numbers should be compared with

- **Majority class** — always answer the most common tag. With six balanced tags this gets 1 in 6 right, so
  anything near 17% means the model learned nothing.
- **Keyword rules** — count words like *pay*, *terminat*, *indemnif* and pick the tag with the most hits. This is
  what a clerk could build with a search box. It does very well on the main test split (the clauses use standard
  legal wording) but collapses on the challenge set, where the wording is plain English.
- **GPT-2 before LoRA training** — the model with the new tag tokens added but nothing trained. This shows how
  much of the final score comes from the LoRA training itself.

The real test for this project is the **challenge column**: beating the keyword rules there shows the model
picked up meaning, not just keywords.

## Limits: where it falls short

- **The dataset is tiny and I wrote it myself.** 72 training/test clauses and 18 challenge clauses are enough to
  show the method works, not to trust it on real records. Scores on 18 test clauses move about 5.6 points per
  clause, so treat differences of one or two clauses as noise.
- **One tag per clause.** A real clause can be about two things (*"pay within 30 days, failing which the
  Authority may terminate"*). The model must pick one.
- **Only six tags.** Anything else (warranties, force majeure, assignment, ...) is forced into the nearest of the
  six. There is no "other" tag yet.
- **Clause splitting is rule-based.** It handles blank lines and numbering like `1.`, `2.1.` and `(a)`, but not
  scanned PDFs, tables, or clauses that run across pages.
- **English only, and GPT-2 is small and old.** Hindi or Kannada records, or very long clauses (over ~1,000
  tokens), are out of scope.
- **Not legal advice.** It is a sorting aid that tells a clerk where to look, not a decision-maker.

## Project layout

```
app/core.py                   all six milestones + baselines + the results report
app/cli.py                    commands: dataset, train, tag, evaluate
corpus/legal_clauses.jsonl    72 labelled clauses (train/test)
corpus/challenge_clauses.jsonl 18 keyword-free clauses (evaluation only)
examples/sample_contract.txt  a document to tag
tests/conftest.py             a tiny GPT-2 built offline, so tests never download anything
tests/test_core.py            one test per milestone + baselines + full pipeline
tests/test_smoke.py           CLI and file-layout checks
results/                      metrics.json, RESULTS.md, tagged_sample.txt (written by `train`)
.github/workflows/ci.yml      runs pytest on every push
.github/workflows/train.yml   trains the real GPT-2 and commits the results
```

## Tests

`pytest` runs 15 tests in about 10 seconds. They use a tiny randomly-initialised GPT-2 (same architecture, toy
size) so they run offline. Among other things they check that:
- LoRA with B = 0 leaves the model's output unchanged, and the layer matches the formula by hand;
- only the LoRA weights change during training, and the loss goes down;
- merging the adapter gives the same predictions as before merging;
- the metrics (precision, recall, F1, confusion matrix) are right on a hand-worked example;
- the split is stratified, has no overlap, and is the same every run.

## Milestones

- [x] 1. Prepare a dataset of legal clauses for tagging
- [x] 2. Extend GPT-2 tokenizer and model for custom tags
- [x] 3. Implement LoRA adapter and inject into GPT-2
- [x] 4. Fine-tune GPT-2 with LoRA on the clause dataset
- [x] 5. Merge LoRA adapter and perform inference on new clauses
- [x] 6. Evaluate tagging accuracy and model efficiency

## Try next

- Rank and layers: `python -m app.cli train --rank 4 --targets c_attn` vs `--rank 16 --targets c_attn c_proj c_fc`.
- Add an `other` tag and a few hundred real clauses, for example from the public LEDGAR contract dataset.
- Allow more than one tag per clause (score each tag separately instead of picking one).
