"""The public tree does not carry private names or a pickled model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "dist", "build"}


def _needles():
    return (
        "certify_" + "cell",
        "q_lb_" + "certified",
        "/Vol" + "umes/",
        "import pick" + "le",
        "import job" + "lib",
        "allow_pickle=" + "True",
    )


def _hits(root):
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in {".npz", ".pyc"}:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in _needles():
            if needle in body:
                hits.append(f"{path.relative_to(root)}: {needle}")
    return hits


class TestTheTreeStaysPublic:
    def test_banned_strings_are_absent(self):
        # GIVEN the package tree, aside from the environment and caches
        hits = _hits(ROOT)
        # THEN none of the private names, the enclave prefix, or a pickle import is present
        assert hits == []

    def test_a_file_that_is_not_utf8_is_skipped(self, tmp_path):
        # GIVEN a tree with a Finder file that is not UTF-8 and a text file that holds a needle
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1\xff\xfe\x80")
        needle = _needles()[2]
        (tmp_path / "notes.txt").write_text(f"rows under {needle}\n", encoding="utf-8")
        # WHEN the tree is scanned
        hits = _hits(tmp_path)
        # THEN the Finder file is skipped and the needle in the text file is still found
        assert hits == [f"notes.txt: {needle}"]
