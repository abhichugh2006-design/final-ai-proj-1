"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import json
import re
import time
from pathlib import Path

DESCRIPTION = ("Tag the clauses of legal documents (confidentiality, termination, payment, ...) with GPT-2 "
               "fine-tuned through a from-scratch LoRA adapter.")
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _cmd_dataset(args) -> None:
    from app import core
    train, test = core.prepare_dataset(args.corpus, seed=args.seed)
    print(f"{len(train)} training clauses, {len(test)} test clauses")
    for label in core.LABELS:
        print(f"  {label:<26} train {sum(e.label == label for e in train):>3}   test {sum(e.label == label for e in test):>3}")
    print("\nExample training string:\n" + core.format_example(train[0]))


def _cmd_train(args) -> None:
    from app import core
    tokenizer, model = core.load_base(args.model)
    started = time.time()
    result = core.run_pipeline(tokenizer, model, args.corpus, args.challenge, r=args.rank, alpha=args.alpha,
                               targets=tuple(args.targets), epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size, seed=args.seed)
    result["config"]["base_model"] = args.model
    result["seconds"] = round(time.time() - started, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "tag_ids.json").write_text(json.dumps(result["tag_ids"], indent=2))

    sample = core.tag_document(model, tokenizer, Path(args.sample).read_text(encoding="utf-8"), result["tag_ids"])
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "metrics.json").write_text(json.dumps(result, indent=2))
    (results / "tagged_sample.txt").write_text(core.render_tagged(sample) + "\n")
    report = core.results_markdown(result, sample)
    (results / "RESULTS.md").write_text("# Results\n\n" + report + "\n")
    print("\n" + report)
    if args.update_readme:
        _update_readme(Path(args.update_readme), report)
    print(f"\nModel saved to {out}/, results to {results}/ ({result['seconds']}s)")


def _update_readme(path: Path, report: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{path} has no {README_START} ... {README_END} block")
    path.write_text(pattern.sub(lambda _: f"{README_START}\n{report}\n{README_END}", text), encoding="utf-8")
    print(f"Updated the results section of {path}")


def _load_trained(path: str):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    if not Path(path, "tag_ids.json").exists():
        raise SystemExit(f"No trained model in {path}/. Run `python -m app.cli train` first.")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).eval()
    tag_ids = json.loads(Path(path, "tag_ids.json").read_text())
    return tokenizer, model, tag_ids


def _cmd_tag(args) -> None:
    from app import core
    tokenizer, model, tag_ids = _load_trained(args.model_dir)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        raise SystemExit("Give a file (--file) or some text (--text).")
    rows = core.tag_document(model, tokenizer, text, tag_ids)
    rendered = json.dumps(rows, indent=2) if args.json else core.render_tagged(rows)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


def _cmd_evaluate(args) -> None:
    from app import core
    tokenizer, model, tag_ids = _load_trained(args.model_dir)
    train, test = core.prepare_dataset(args.corpus, seed=args.seed)
    sets = {"test": test}
    if Path(args.challenge).exists():
        sets["challenge"] = core.load_clauses(args.challenge)
    for name, examples in sets.items():
        rep = core.evaluate(model, tokenizer, examples, tag_ids)
        base = core.baselines(train, examples)
        print(f"{name}: LoRA accuracy {rep['accuracy']:.1%} (macro-F1 {rep['macro_f1']:.3f}) | "
              f"keyword rules {base['keyword_rules']['accuracy']:.1%} | majority {base['majority_class']['accuracy']:.1%}")
        for m in rep["mistakes"]:
            print(f"   expected {m['true']}, got {m['pred']}: {m['text'][:90]}")


def build_parser() -> argparse.ArgumentParser:
    from app import core
    parser = argparse.ArgumentParser(prog="legal-document-lora-tagger", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dataset", help="milestone 1: show the clause dataset and the train/test split")
    p.add_argument("--corpus", default=str(core.DEFAULT_CORPUS))
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="milestones 2-6: add tag tokens, inject LoRA, fine-tune, merge, evaluate, save")
    p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--corpus", default=str(core.DEFAULT_CORPUS))
    p.add_argument("--challenge", default=str(core.DEFAULT_CHALLENGE))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--targets", nargs="+", default=["c_attn", "c_proj"], help="GPT-2 layers to adapt: c_attn c_proj c_fc")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/lora-tagger")
    p.add_argument("--results", default="results")
    p.add_argument("--sample", default="examples/sample_contract.txt")
    p.add_argument("--update-readme", metavar="README.md", help="also write the results table into this README")

    p = sub.add_parser("tag", help="milestone 5: tag every clause in a document")
    p.add_argument("--file")
    p.add_argument("--text")
    p.add_argument("--model-dir", default="models/lora-tagger")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")

    p = sub.add_parser("evaluate", help="milestone 6: score the saved model against both baselines")
    p.add_argument("--model-dir", default="models/lora-tagger")
    p.add_argument("--corpus", default=str(core.DEFAULT_CORPUS))
    p.add_argument("--challenge", default=str(core.DEFAULT_CHALLENGE))
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"dataset": _cmd_dataset, "train": _cmd_train, "tag": _cmd_tag, "evaluate": _cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
