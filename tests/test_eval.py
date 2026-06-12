"""Тесты eval-слоя на маленькой синтетической выборке (без датасета и сети)."""

from collections import Counter

import pytest

from eval.baseline import predict, train_baseline
from eval.classify import ClassifyResult, classify_resume, normalize_label
from eval.metrics import compute_metrics, save_confusion_matrix
from eval.prepare_data import Split, load_dataset, make_split
from eval.run_metrics import run_baseline
from llm.mock_client import MockLLMClient
from llm.prompt_loader import load_classify_prompt

# Две легко разделимые категории по словарю.
_BACKEND = [f"python django flask backend api sql postgres server {i}" for i in range(8)]
_FRONTEND = [f"react vue javascript css html frontend ui browser {i}" for i in range(8)]
_TEXTS = _BACKEND + _FRONTEND
_LABELS = ["Backend"] * 8 + ["Frontend"] * 8


def _synthetic_split() -> Split:
    return make_split(_TEXTS, _LABELS, test_size=0.25, seed=0)


# --- metrics --------------------------------------------------------------


def test_compute_metrics_known_values():
    m = compute_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert m["accuracy"] == 0.75
    assert m["macro_f1"] == pytest.approx(0.7333333, abs=1e-6)  # именно macro, не micro(=0.75)


def test_compute_metrics_offlist_pred_no_phantom_class():
    # "x" вне gold-классов {a,b}: штрафуется как потеря recall, но НЕ как отдельный
    # класс с F1=0 -> macro по {a,b}, а не по {a,b,x}
    m = compute_metrics(["a", "a", "b"], ["a", "x", "b"])
    assert m["macro_f1"] == pytest.approx(0.8333333, abs=1e-6)
    assert compute_metrics(["a", "a", "b"], ["a", "x", "b"], labels=["a", "b"]) == m


def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(["a"], ["a", "b"])


def test_compute_metrics_empty():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_save_confusion_matrix_writes_png(tmp_path):
    out = save_confusion_matrix(
        ["a", "b", "a"], ["a", "b", "b"], ["a", "b"], tmp_path / "cm.png"
    )
    assert out.exists() and out.stat().st_size > 0


# --- baseline -------------------------------------------------------------


def test_baseline_trains_and_predicts_valid_labels():
    split = _synthetic_split()
    model = train_baseline(split.train_texts, split.train_labels)
    preds = predict(model, split.test_texts)
    assert len(preds) == len(split.test_texts)
    assert set(preds) <= {"Backend", "Frontend"}
    # словарь классов не пересекается -> baseline должен хорошо разделять
    assert compute_metrics(split.test_labels, preds)["accuracy"] >= 0.5


def test_run_baseline_end_to_end_writes_cm(tmp_path):
    model, metrics = run_baseline(_synthetic_split(), tmp_path)
    assert model is not None
    assert "accuracy" in metrics and "macro_f1" in metrics
    assert (tmp_path / "cm_baseline.png").exists()


# --- prepare_data ---------------------------------------------------------


def test_make_split_sized_and_labels():
    split = _synthetic_split()
    assert len(split.test_texts) == 4  # 25% от 16
    assert len(split.train_texts) == 12
    assert split.labels == ["Backend", "Frontend"]


def test_make_split_preserves_class_proportions():
    # имбаланс 9:3 -> стратификация сохраняет пропорции (в тесте 3:1).
    # На сбалансированных данных тест прошёл бы и без stratify — поэтому имбаланс.
    texts = [f"text {i}" for i in range(12)]
    labels = ["A"] * 9 + ["B"] * 3
    split = make_split(texts, labels, test_size=1 / 3, seed=0)
    assert Counter(split.test_labels) == {"A": 3, "B": 1}
    assert Counter(split.train_labels) == {"A": 6, "B": 2}


def test_load_dataset_autodetect_and_clean(tmp_path):
    csv = tmp_path / "resumes.csv"
    csv.write_text(
        "Resume,Category\n"
        "python backend,Backend\n"
        "react frontend,Frontend\n"
        " ,Backend\n",  # пустой текст -> отбрасывается
        encoding="utf-8",
    )
    texts, labels = load_dataset(str(csv))
    assert texts == ["python backend", "react frontend"]
    assert labels == ["Backend", "Frontend"]


def test_load_dataset_missing_column_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(str(csv))


# --- classify-промпт и LLM-классификация ----------------------------------


def test_load_classify_prompt_has_placeholders():
    p = load_classify_prompt()
    assert p.name == "classify_resume"
    assert "{categories}" in p.user_template
    assert "{resume}" in p.user_template


def test_normalize_label_case_insensitive():
    cats = ["Data Science", "HR"]
    assert normalize_label("data science", cats) == "Data Science"
    assert normalize_label("  HR ", cats) == "HR"
    assert normalize_label("Unknown", cats) == "Unknown"  # вне списка -> как есть


def test_classify_result_schema():
    assert ClassifyResult.model_validate_json('{"category": "HR"}').category == "HR"


async def test_classify_resume_returns_normalized_label():
    client = MockLLMClient(['{"category": "data science"}'])  # неточный регистр
    prompt = load_classify_prompt()
    label = await classify_resume(client, prompt, ["Data Science", "HR"], "резюме")
    assert label == "Data Science"
