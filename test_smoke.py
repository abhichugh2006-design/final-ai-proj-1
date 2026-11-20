"""Plumbing checks: the package imports, the CLI parses, the data files are where the README says."""
from pathlib import Path

import pytest

from app import core
from app.cli import build_parser, main


def test_readme_files_exist():
    for path in ["corpus/legal_clauses.jsonl", "corpus/challenge_clauses.jsonl", "examples/sample_contract.txt",
                 "requirements.txt", ".github/workflows/ci.yml", ".github/workflows/train.yml"]:
        assert Path(path).exists(), path


def test_every_readme_command_parses():
    parser = build_parser()
    for cmd in (["dataset"], ["train", "--rank", "4", "--targets", "c_attn", "c_proj"],
                ["train", "--update-readme", "README.md"],
                ["tag", "--file", "examples/sample_contract.txt"], ["evaluate"]):
        assert parser.parse_args(cmd).command == cmd[0]


def test_dataset_command_prints_the_split(capsys):
    main(["dataset"])
    out = capsys.readouterr().out
    assert "54 training clauses, 18 test clauses" in out
    assert "<|tag:" in out


def test_tag_without_a_model_explains_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="Run `python -m app.cli train` first"):
        main(["tag", "--text", "Pay within 30 days.", "--model-dir", str(tmp_path)])


def test_readme_has_a_results_block():
    text = Path("README.md").read_text()
    assert "<!-- RESULTS:START -->" in text and "<!-- RESULTS:END -->" in text
    assert core.LABELS
