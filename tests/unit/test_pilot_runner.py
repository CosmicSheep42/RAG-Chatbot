import runpy
from pathlib import Path


REFUSAL_PATTERN = runpy.run_path(
    Path(__file__).resolve().parents[2] / "evaluation/pilots/full-rag-gpu-dev/run_pilot.py"
)["REFUSAL_PATTERN"]


def test_pilot_refusal_heuristic_detects_safety_refusal_wording():
    assert REFUSAL_PATTERN.search("I can't provide information on harmful or illegal content.")
    assert not REFUSAL_PATTERN.search("If you can't provide one or more components, you cannot cast the spell.")
