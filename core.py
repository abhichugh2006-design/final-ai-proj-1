"""Legal Document LoRA Tagger: one section per milestone, built in order.

The idea in one line: GPT-2 reads a clause and predicts ONE special tag token
(e.g. ``<|tag:termination|>``). We teach it that skill by training only a small
LoRA adapter on the attention layers, then merge the adapter back in.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Milestone 1: Prepare a dataset of legal clauses for tagging
# ---------------------------------------------------------------------------

LABELS: list[str] = [
    "confidentiality",
    "termination",
    "indemnification",
    "payment",
    "governing_law",
    "limitation_of_liability",
]

# Plain-English words used to give each new tag token a sensible starting embedding (milestone 2).
LABEL_HINTS: dict[str, str] = {
    "confidentiality": " confidential secret disclose",
    "termination": " terminate end cancel",
    "indemnification": " indemnify compensate harmless",
    "payment": " pay invoice fee",
    "governing_law": " law court jurisdiction",
    "limitation_of_liability": " liability limit damages",
}

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "legal_clauses.jsonl"
PROMPT_TEMPLATE = "Clause: {text}\nTag:"


def tag_token(label: str) -> str:
    """The special token that stands for one label, e.g. ``<|tag:payment|>``."""
    return f"<|tag:{label}|>"


@dataclass
class Example:
    text: str
    label: str


def load_clauses(path: str | Path = DEFAULT_CORPUS) -> list[Example]:
    """Read ``{"text": ..., "label": ...}`` lines from a JSONL file and check every label is known."""
    examples = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text, label = " ".join(str(row["text"]).split()), str(row["label"]).strip()
        if label not in LABELS:
            raise ValueError(f"line {n}: unknown label {label!r}; expected one of {LABELS}")
        if text:
            examples.append(Example(text, label))
    return examples


def format_prompt(text: str) -> str:
    """What the model sees at inference time; it must answer with one tag token."""
    return PROMPT_TEMPLATE.format(text=" ".join(text.split()))


def format_example(ex: Example) -> str:
    """What the model sees during training: prompt + the answer tag + end of text."""
    return f"{format_prompt(ex.text)}{tag_token(ex.label)}<|endoftext|>"


def train_test_split(examples: list[Example], test_fraction: float = 0.25,
                     seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Stratified split: every label keeps the same share in train and test. Deterministic for a given seed."""
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    train, test = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction)) if len(group) > 1 else 0
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    return train, test


def prepare_dataset(path: str | Path = DEFAULT_CORPUS, test_fraction: float = 0.25, seed: int = 0):
    """Load the corpus and return (train, test) examples."""
    return train_test_split(load_clauses(path), test_fraction, seed)


# ---------------------------------------------------------------------------
# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
# ---------------------------------------------------------------------------

def add_tag_tokens(tokenizer, model, labels: list[str] = LABELS) -> dict[str, int]:
    """Add one special token per label, resize the embeddings, and return {label: token_id}.

    New rows start as the average embedding of a few hint words (instead of random noise),
    so the model already "leans" the right way before any training. GPT-2 ties its input
    embeddings to the output head, so this also sets the output scores for each tag.
    """
    tokens = [tag_token(label) for label in labels]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    emb = model.get_input_embeddings().weight
    tag_ids = {}
    with torch.no_grad():
        for label, tok in zip(labels, tokens):
            tid = tokenizer.convert_tokens_to_ids(tok)
            hint_ids = tokenizer(LABEL_HINTS.get(label, " " + label.replace("_", " ")))["input_ids"]
            hint_ids = [i for i in hint_ids if i < emb.shape[0] and i != tid]
            if hint_ids:
                emb[tid] = emb[hint_ids].mean(dim=0)
            tag_ids[label] = tid
    return tag_ids


def load_base(model_name: str = "gpt2"):
    """Download (first time only) the pretrained GPT-2 and its tokenizer."""
    from transformers import AutoTokenizer, GPT2LMHeadModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model



# ---------------------------------------------------------------------------
# Milestone 3: Implement LoRA adapter and inject into GPT-2
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen layer W and adds a low-rank update:  y = W x + (alpha / r) * B(A(x)).

    A is (r x in) with small random values, B is (out x r) and starts at ZERO, so on
    step 0 the model behaves exactly like the original. Only A and B are trained.

    Works with both ``nn.Linear`` (weight is out x in) and GPT-2's ``Conv1D`` (weight is in x out).
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.is_conv1d = type(base).__name__ == "Conv1D"
        w = base.weight
        self.in_features, self.out_features = (w.shape[0], w.shape[1]) if self.is_conv1d else (w.shape[1], w.shape[0])
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + update * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The full-size weight change B @ A * scale, laid out like the base layer's weight."""
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
        return delta.T if self.is_conv1d else delta

    def merged(self) -> nn.Module:
        """Return the base layer with the LoRA update folded into its weight (no extra cost at inference)."""
        with torch.no_grad():
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
        for p in self.base.parameters():
            p.requires_grad = True
        return self.base


def inject_lora(model: nn.Module, target_modules: tuple[str, ...] = ("c_attn",),
                r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Freeze the whole model, then swap every target layer for a LoRALinear. Returns the replaced names.

    GPT-2 names: ``c_attn`` = fused Q/K/V projection, ``c_proj`` = attention/MLP output, ``c_fc`` = MLP input.
    """
    for p in model.parameters():
        p.requires_grad = False
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_modules and not isinstance(child, LoRALinear) and hasattr(child, "weight"):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    if not replaced:
        raise ValueError(f"no modules named {target_modules} found")
    return replaced


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / max(total, 1)}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: this is the small file you would ship or share."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}



# ---------------------------------------------------------------------------
# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
# ---------------------------------------------------------------------------

def encode_batch(tokenizer, examples: list[Example], max_length: int = 128) -> dict[str, torch.Tensor]:
    """Tokenise training strings and set labels to -100 everywhere except the tag (and end) tokens,
    so the loss only measures "did you pick the right tag", not "can you copy the clause"."""
    ids_list, label_list = [], []
    for ex in examples:
        prompt_ids = tokenizer(format_prompt(ex.text))["input_ids"]
        answer_ids = [tokenizer.convert_tokens_to_ids(tag_token(ex.label)), tokenizer.eos_token_id]
        prompt_ids = prompt_ids[: max(1, max_length - len(answer_ids))]
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        ids_list.append(ids)
        label_list.append(labels)
    width = max(len(x) for x in ids_list)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(ids_list), width), pad, dtype=torch.long)
    labels = torch.full((len(ids_list), width), -100, dtype=torch.long)
    attention = torch.zeros((len(ids_list), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(ids_list, label_list)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def train(model, tokenizer, examples: list[Example], epochs: int = 8, lr: float = 2e-3,
          batch_size: int = 8, seed: int = 0, log=print) -> list[float]:
    """Plain PyTorch loop over the LoRA parameters only. Returns the mean loss of each epoch."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nothing to train: call inject_lora() first")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device
    history = []
    model.train()
    for epoch in range(epochs):
        order = examples[:]
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = {k: v.to(device) for k, v in encode_batch(tokenizer, order[start:start + batch_size]).items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append(sum(losses) / len(losses))
        if log:
            log(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")
    model.eval()
    return history



# ---------------------------------------------------------------------------
# Milestone 5: Merge LoRA adapter and perform inference on new clauses
# ---------------------------------------------------------------------------

def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear with its base layer, with B@A folded into the weight. Returns how many merged."""
    merged = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
                merged += 1
    model.eval()
    return merged


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], tag_ids: dict[str, int],
            batch_size: int = 16) -> list[tuple[str, float]]:
    """For each clause, look at the next-token scores and pick the best TAG token (constrained decoding).

    Returns (label, confidence) pairs; confidence is a softmax over the tag tokens only.
    """
    labels = list(tag_ids)
    ids_tensor = torch.tensor([tag_ids[l] for l in labels])
    device = next(model.parameters()).device
    results = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        chunk = [tokenizer(format_prompt(t))["input_ids"][-1000:] for t in texts[start:start + batch_size]]
        width = max(len(c) for c in chunk)
        # left-pad so the last real token is always at the final position
        input_ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, c in enumerate(chunk):
            input_ids[i, width - len(c):] = torch.tensor(c)
            attention[i, width - len(c):] = 1
        position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                       position_ids=position_ids.to(device)).logits[:, -1, :]
        probs = logits[:, ids_tensor.to(device)].softmax(-1).cpu()
        for row in probs:
            k = int(row.argmax())
            results.append((labels[k], float(row[k])))
    return results


def split_clauses(document: str) -> list[str]:
    """Split a document into clauses: blank-line paragraphs, then numbered items like "1." or "(a)".

    Headings (short lines in capitals) are dropped.
    """
    parts = []
    for para in re.split(r"\n\s*\n", document.strip()):
        para = " ".join(para.split())
        for piece in re.split(r"(?:(?<=\s)|^)(?=(?:\d+(?:\.\d+)*\.|\([a-z0-9]+\))\s)", para):
            piece = piece.strip()
            body = re.sub(r"^(?:\d+(?:\.\d+)*\.|\([a-z0-9]+\))\s*", "", piece)
            if len(body.split()) >= 4 and not (body.isupper() and len(body.split()) < 15):
                parts.append(body)
    return parts


def tag_document(model, tokenizer, document: str, tag_ids: dict[str, int]) -> list[dict]:
    """Split a document and tag every clause. Returns [{"clause", "tag", "confidence"}]."""
    clauses = split_clauses(document)
    preds = predict(model, tokenizer, clauses, tag_ids) if clauses else []
    return [{"clause": c, "tag": t, "confidence": round(p, 4)} for c, (t, p) in zip(clauses, preds)]


def render_tagged(rows: list[dict]) -> str:
    """Human-readable tagged document: one clause per block with its tag in front."""
    return "\n\n".join(f"[{r['tag'].upper()}] ({r['confidence']:.0%})\n{r['clause']}" for r in rows)



# ---------------------------------------------------------------------------
# Milestone 6: Evaluate tagging accuracy and model efficiency
# ---------------------------------------------------------------------------

def classification_report(y_true: list[str], y_pred: list[str], labels: list[str] = LABELS) -> dict:
    """Accuracy, per-label precision/recall/F1, macro-F1 and a confusion matrix (rows = truth)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    per_label = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    present = [l for l in labels if per_label[l]["support"] > 0]
    confusion = {t: dict(Counter(p for tt, p in zip(y_true, y_pred) if tt == t)) for t in labels}
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1),
        "macro_f1": sum(per_label[l]["f1"] for l in present) / max(len(present), 1),
        "per_label": per_label,
        "confusion": confusion,
        "n": len(y_true),
    }


def efficiency_report(model: nn.Module, full_model_params: int | None = None) -> dict:
    """How much smaller the LoRA job is than full fine-tuning (call this BEFORE merging)."""
    counts = count_parameters(model)
    full = full_model_params or counts["total"]
    adapter = sum(v.numel() for v in lora_state_dict(model).values())
    return {
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "trainable_pct": round(counts["trainable_pct"], 4),
        "adapter_params": adapter,
        "adapter_mb_fp32": round(adapter * 4 / 1e6, 3),
        "full_finetune_mb_fp32": round(full * 4 / 1e6, 1),
        "reduction_factor": round(full / max(counts["trainable"], 1), 1),
    }


def evaluate(model, tokenizer, examples: list[Example], tag_ids: dict[str, int]) -> dict:
    preds = predict(model, tokenizer, [e.text for e in examples], tag_ids)
    report = classification_report([e.label for e in examples], [p for p, _ in preds], list(tag_ids))
    report["mistakes"] = [{"text": e.text, "true": e.label, "pred": p}
                          for e, (p, _) in zip(examples, preds) if p != e.label]
    return report



# ---------------------------------------------------------------------------
# Baselines: what you get WITHOUT LoRA, so the LoRA numbers mean something
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, list[str]] = {
    "confidentiality": ["confidential", "disclos", "secre", "non-public", "privileged"],
    "termination": ["terminat", "cancel", "rescind", "end", "expiry", "notice"],
    "indemnification": ["indemnif", "hold harmless", "harmless", "defend", "reimburse", "make good"],
    "payment": ["pay", "invoice", "fee", "instalment", "price", "rent", "interest"],
    "governing_law": ["law", "court", "jurisdiction", "arbitrat", "governed"],
    "limitation_of_liability": ["liable", "liability", "damages", "not exceed", "capped", "disclaim"],
}


def keyword_tag(text: str) -> str:
    """A hand-written rule tagger: the label whose keywords appear most often wins (ties: first label)."""
    low = text.lower()
    scores = {label: sum(low.count(k) for k in words) for label, words in KEYWORDS.items()}
    return max(LABELS, key=lambda l: scores[l])


def baselines(train_set: list[Example], test_set: list[Example]) -> dict:
    """Majority-class guess and the keyword rules, scored on the same test split."""
    majority = Counter(e.label for e in train_set).most_common(1)[0][0]
    truth = [e.label for e in test_set]
    out = {}
    for name, preds in {"majority_class": [majority] * len(test_set),
                        "keyword_rules": [keyword_tag(e.text) for e in test_set]}.items():
        rep = classification_report(truth, preds)
        out[name] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}
    return out


# ---------------------------------------------------------------------------
# The whole pipeline, used by the command line
# ---------------------------------------------------------------------------

DEFAULT_CHALLENGE = DEFAULT_CORPUS.parent / "challenge_clauses.jsonl"


def _score(model, tokenizer, examples, tag_ids, train_set) -> dict:
    """Model + both baselines on one evaluation set."""
    report = evaluate(model, tokenizer, examples, tag_ids)
    return {"lora": report, "baselines": baselines(train_set, examples)}


def run_pipeline(tokenizer, model, corpus: str | Path = DEFAULT_CORPUS,
                 challenge: str | Path | None = DEFAULT_CHALLENGE, r: int = 8, alpha: float = 16.0,
                 targets: tuple[str, ...] = ("c_attn", "c_proj"), epochs: int = 10, lr: float = 1e-3,
                 batch_size: int = 8, seed: int = 0, log=print) -> dict:
    """Milestones 1-6 end to end. Returns metrics; the model is left merged and ready for inference."""
    torch.manual_seed(seed)
    train_set, test_set = prepare_dataset(corpus, seed=seed)
    challenge_set = load_clauses(challenge) if challenge and Path(challenge).exists() else []
    tag_ids = add_tag_tokens(tokenizer, model)
    zero_shot = evaluate(model, tokenizer, test_set, tag_ids)
    inject_lora(model, targets, r=r, alpha=alpha)
    efficiency = efficiency_report(model)
    history = train(model, tokenizer, train_set, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, log=log)
    merge_lora(model)
    result = {
        "config": {"base_model": getattr(model.config, "_name_or_path", "") or "gpt2", "r": r, "alpha": alpha,
                   "targets": list(targets), "epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
                   "train_size": len(train_set), "test_size": len(test_set), "challenge_size": len(challenge_set)},
        "loss_history": history,
        "zero_shot_test": {"accuracy": zero_shot["accuracy"], "macro_f1": zero_shot["macro_f1"]},
        "test": _score(model, tokenizer, test_set, tag_ids, train_set),
        "efficiency": efficiency,
        "tag_ids": tag_ids,
    }
    if challenge_set:
        result["challenge"] = _score(model, tokenizer, challenge_set, tag_ids, train_set)
    return result


def results_markdown(result: dict, tagged_sample: list[dict] | None = None) -> str:
    """Turn a run's metrics into the Markdown table that goes in README.md and results/RESULTS.md."""
    c, eff = result["config"], result["efficiency"]
    lines = [f"Base model `{c['base_model']}`, LoRA r={c['r']}, alpha={c['alpha']:g} on `{' '.join(c['targets'])}`, "
             f"{c['epochs']} epochs, lr {c['lr']:g}, seed {c['seed']}. Train {c['train_size']} / test {c['test_size']}"
             + (f" / challenge {c['challenge_size']}" if c.get("challenge_size") else "") + " clauses.", "",
             "| Method | Test accuracy | Test macro-F1 |" + (" Challenge accuracy | Challenge macro-F1 |" if "challenge" in result else ""),
             "|---|---|---|" + ("---|---|" if "challenge" in result else "")]

    def row(name, t, ch=None):
        cells = f"| {name} | {t['accuracy']:.1%} | {t['macro_f1']:.3f} |"
        if "challenge" in result:
            cells += f" {ch['accuracy']:.1%} | {ch['macro_f1']:.3f} |" if ch else " – | – |"
        return cells

    tb, cb = result["test"]["baselines"], result.get("challenge", {}).get("baselines", {})
    lines.append(row("Majority class (always the same tag)", tb["majority_class"], cb.get("majority_class")))
    lines.append(row("Keyword rules (`keyword_tag`)", tb["keyword_rules"], cb.get("keyword_rules")))
    lines.append(row("GPT-2 + tag tokens, before LoRA training", result["zero_shot_test"]))
    lines.append(row("**GPT-2 + LoRA (this project)**", result["test"]["lora"], result.get("challenge", {}).get("lora")))
    lines += ["", f"**Efficiency:** {eff['trainable_params']:,} trainable parameters out of {eff['total_params']:,} "
              f"({eff['trainable_pct']:.2f}%), {eff['reduction_factor']:g}x fewer than full fine-tuning. "
              f"The adapter is {eff['adapter_mb_fp32']} MB vs {eff['full_finetune_mb_fp32']} MB for a full copy of the model. "
              "After merging, inference costs exactly the same as plain GPT-2.", "",
              f"**Training loss by epoch:** {' → '.join(f'{x:.3f}' for x in result['loss_history'])}", "",
              "**Per-label F1 (test / challenge):**", ""]
    for label in LABELS:
        t = result["test"]["lora"]["per_label"][label]["f1"]
        ch = result.get("challenge", {}).get("lora", {}).get("per_label", {}).get(label, {}).get("f1")
        lines.append(f"- `{label}`: {t:.2f}" + (f" / {ch:.2f}" if ch is not None else ""))
    mistakes = result["test"]["lora"]["mistakes"] + result.get("challenge", {}).get("lora", {}).get("mistakes", [])
    if mistakes:
        lines += ["", "**Where it went wrong:**", ""]
        lines += [f"- expected `{m['true']}`, got `{m['pred']}`: \"{m['text']}\"" for m in mistakes[:8]]
    if tagged_sample:
        lines += ["", "**Sample output** (`examples/sample_contract.txt`):", "", "```", render_tagged(tagged_sample), "```"]
    return "\n".join(lines)
