from evaluation.adapters.datapizza import load_examples
from evaluation.smoke.gpu.run_smoke import OUTPUT, QUESTION_ID, ROOT


def test_gpu_smoke_is_limited_to_one_development_question_and_separate_output():
    examples = load_examples(ROOT / "evaluation/dataset/datapizza", "dev")

    assert [example.id for example in examples if example.id == QUESTION_ID] == [QUESTION_ID]
    assert OUTPUT == ROOT / "evaluation/smoke/gpu/results.json"
    assert "published" not in OUTPUT.parts
