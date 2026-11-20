"""One test per milestone (plus extras for the LoRA math, clause splitting, baselines and the full pipeline).
They use the tiny offline GPT-2 from conftest.py, so nothing is downloaded."""
import json

import pytest
import torch

from app import core


# Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1_dataset(tmp_path):
    train, test = core.prepare_dataset()
    everything = train + test
    assert len(everything) == 72
    assert {e.label for e in everything} == set(core.LABELS)
    assert {e.label for e in test} == set(core.LABELS)          # stratified: every label is tested
    assert not {e.text for e in train} & {e.text for e in test}  # no leakage
    assert core.prepare_dataset()[1] == test                          # deterministic

    text = core.format_example(core.Example("Pay within 30 days.", "payment"))
    assert text == "Clause: Pay within 30 days.\nTag:<|tag:payment|><|endoftext|>"

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "x y z", "label": "not_a_label"}) + "\n")
    with pytest.raises(ValueError):
        core.load_clauses(bad)


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2_tag_tokens(tiny):
    tokenizer, model = tiny
    before = len(tokenizer)
    tag_ids = core.add_tag_tokens(tokenizer, model)
    assert set(tag_ids) == set(core.LABELS)
    assert len(tokenizer) == before + len(core.LABELS)
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    # each tag is ONE token, not split into pieces
    ids = tokenizer("Tag:<|tag:payment|>")["input_ids"]
    assert ids[-1] == tag_ids["payment"]
    assert tokenizer.pad_token is not None


# Milestone 3: Implement LoRA adapter and inject into GPT-2
def test_milestone_3_lora_injection(tiny):
    tokenizer, model = tiny
    core.add_tag_tokens(tokenizer, model)
    x = torch.tensor([tokenizer("Clause: hello\nTag:")["input_ids"]])
    with torch.no_grad():
        before = model(x).logits
    replaced = core.inject_lora(model, ("c_attn",), r=4, alpha=8)
    assert len(replaced) == model.config.n_layer
    with torch.no_grad():
        after = model(x).logits
    assert torch.allclose(before, after, atol=1e-6)  # B starts at zero, so nothing changes yet
    counts = core.count_parameters(model)
    expected = model.config.n_layer * 4 * (model.config.n_embd + 3 * model.config.n_embd)
    assert counts["trainable"] == expected
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)


def test_lora_layer_matches_manual_math():
    base = torch.nn.Linear(5, 3)
    layer = core.LoRALinear(base, r=2, alpha=4)
    torch.nn.init.normal_(layer.lora_B)
    x = torch.randn(4, 5)
    expected = base(x) + (x @ layer.lora_A.T @ layer.lora_B.T) * 2.0
    assert torch.allclose(layer(x), expected, atol=1e-6)


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def test_milestone_4_finetune(tiny):
    tokenizer, model = tiny
    train, _ = core.prepare_dataset()
    core.add_tag_tokens(tokenizer, model)
    core.inject_lora(model, ("c_attn", "c_proj"), r=8, alpha=16)
    frozen = model.transformer.wte.weight.clone()
    batch = core.encode_batch(tokenizer, train[:2])
    assert (batch["labels"] != -100).sum() == 4  # only tag + end token are scored, per example
    history = core.train(model, tokenizer, train, epochs=6, lr=1e-2, log=None)
    assert history[-1] < history[0]
    assert torch.equal(model.transformer.wte.weight, frozen)  # base weights untouched


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
def test_milestone_5_merge_and_tag(tiny):
    tokenizer, model = tiny
    train, _ = core.prepare_dataset()
    tag_ids = core.add_tag_tokens(tokenizer, model)
    core.inject_lora(model, ("c_attn",), r=4, alpha=8)
    core.train(model, tokenizer, train[:16], epochs=1, lr=1e-2, log=None)
    texts = ["The Vendor shall pay the fee.", "Either party may terminate this Agreement."]
    unmerged = core.predict(model, tokenizer, texts, tag_ids)
    assert core.merge_lora(model) == model.config.n_layer
    assert not any(isinstance(m, core.LoRALinear) for m in model.modules())
    merged = core.predict(model, tokenizer, texts, tag_ids)
    for (l1, p1), (l2, p2) in zip(unmerged, merged):
        assert l1 == l2 and abs(p1 - p2) < 1e-4  # merging changes nothing about the answers

    doc = open("examples/sample_contract.txt").read()
    rows = core.tag_document(model, tokenizer, doc, tag_ids)
    assert len(rows) == 6
    assert all(r["tag"] in core.LABELS and 0 < r["confidence"] <= 1 for r in rows)
    assert "[" in core.render_tagged(rows)


def test_split_clauses():
    doc = "AGREEMENT TITLE\n\n1. The first clause is here. 2. The second clause is here too.\n\n(a) A lettered clause follows now."
    assert core.split_clauses(doc) == ["The first clause is here.", "The second clause is here too.",
                                       "A lettered clause follows now."]


# Milestone 6: Evaluate tagging accuracy and model efficiency
def test_milestone_6_evaluation(tiny):
    report = core.classification_report(["payment", "payment", "termination"],
                                        ["payment", "termination", "termination"], ["payment", "termination"])
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["per_label"]["payment"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert report["macro_f1"] == pytest.approx(2 / 3)
    assert report["confusion"]["payment"] == {"payment": 1, "termination": 1}

    tokenizer, model = tiny
    _, test = core.prepare_dataset()
    tag_ids = core.add_tag_tokens(tokenizer, model)
    core.inject_lora(model, ("c_attn",), r=4)
    eff = core.efficiency_report(model)
    assert eff["trainable_params"] == eff["adapter_params"] > 0
    assert eff["reduction_factor"] > 1
    result = core.evaluate(model, tokenizer, test, tag_ids)
    assert result["n"] == len(test) and 0 <= result["accuracy"] <= 1


def test_baselines_and_challenge_set():
    train, test = core.prepare_dataset()
    challenge = core.load_clauses(core.DEFAULT_CHALLENGE)
    assert len(challenge) == 18 and {e.label for e in challenge} == set(core.LABELS)
    assert not {e.text for e in challenge} & {e.text for e in train + test}
    assert core.keyword_tag("The Purchaser shall pay the invoice.") == "payment"
    b = core.baselines(train, test)
    assert b["majority_class"]["accuracy"] == pytest.approx(1 / 6)
    assert b["keyword_rules"]["accuracy"] > 0.9                                    # the main split is keyword-easy...
    assert core.baselines(train, challenge)["keyword_rules"]["accuracy"] < 0.5     # ...the challenge set is not


def test_whole_pipeline_runs_and_reports(tiny, tmp_path):
    tokenizer, model = tiny
    out = core.run_pipeline(tokenizer, model, epochs=1, r=4, log=None)
    assert out["test"]["lora"]["n"] == out["config"]["test_size"] == 18
    assert out["challenge"]["lora"]["n"] == 18
    assert len(out["loss_history"]) == 1
    sample = core.tag_document(model, tokenizer, open("examples/sample_contract.txt").read(), out["tag_ids"])
    md = core.results_markdown(out, sample)
    assert "Keyword rules" in md and "GPT-2 + LoRA" in md and "Sample output" in md

    from app.cli import _update_readme
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- RESULTS:START -->\nPLACEHOLDER-XYZ\n<!-- RESULTS:END -->\nafter\n")
    _update_readme(readme, md)
    text = readme.read_text()
    assert "PLACEHOLDER-XYZ" not in text and "Keyword rules" in text and text.startswith("before") and text.endswith("after\n")
