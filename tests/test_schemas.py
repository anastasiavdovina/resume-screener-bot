"""Unit-тесты pydantic-схем: корректный и некорректный «ответ модели».

Эти тесты не требуют ни сети, ни ключей — проверяют только контракт данных.
"""

import pytest
from pydantic import ValidationError

from service.schemas import MatchResult, Report, ResumeFields

# --- Валидные данные ------------------------------------------------------


def test_resume_fields_full_valid():
    r = ResumeFields(
        category="Python Developer",
        skills=["Python", "Docker", "PostgreSQL"],
        years_experience=4.5,
        grade="middle",
        education="БГУ, прикладная математика",
        summary="Бэкенд-разработчик с опытом микросервисов.",
        language="ru",
    )
    assert r.grade == "middle"
    assert r.years_experience == 4.5
    assert "Docker" in r.skills


def test_resume_fields_nullable_optionals():
    """Отсутствующие в резюме поля приходят как null — это валидно (анти-галлюцинации)."""
    r = ResumeFields(
        category="Arts",
        summary="Резюме без явного опыта и грейда.",
        language="en",
    )
    assert r.years_experience is None
    assert r.grade is None
    assert r.education is None
    assert r.skills == []  # default_factory


def test_resume_fields_from_json_string():
    """Реалистичный путь: model_validate_json на сыром выводе LLM."""
    raw = (
        '{"category": "Data Science", "skills": ["pandas", "sklearn"], '
        '"years_experience": 2, "grade": "junior", "education": null, '
        '"summary": "DS junior.", "language": "en"}'
    )
    r = ResumeFields.model_validate_json(raw)
    assert r.category == "Data Science"
    assert r.years_experience == 2.0


def test_resume_fields_ignores_extra_keys():
    """LLM добавил лишнее поле — игнорируем, не падаем (extra='ignore')."""
    r = ResumeFields.model_validate(
        {
            "category": "HR",
            "summary": "HR generalist.",
            "language": "en",
            "hallucinated_field": "ignore me",
        }
    )
    assert not hasattr(r, "hallucinated_field")


# --- Невалидные данные ----------------------------------------------------


def test_resume_fields_missing_required_summary():
    with pytest.raises(ValidationError):
        ResumeFields(category="HR", language="ru")  # нет summary


def test_resume_fields_bad_grade():
    with pytest.raises(ValidationError):
        ResumeFields(
            category="HR", summary="x", language="ru", grade="god-tier"  # не из Literal
        )


def test_resume_fields_negative_experience():
    with pytest.raises(ValidationError):
        ResumeFields(
            category="HR", summary="x", language="ru", years_experience=-1
        )


def test_resume_fields_bad_language():
    with pytest.raises(ValidationError):
        ResumeFields(category="HR", summary="x", language="de")  # не ru/en


# --- MatchResult ----------------------------------------------------------


def test_match_result_valid():
    m = MatchResult(
        score=82,
        verdict="match",
        strengths=["Опыт с Docker"],
        gaps=["Нет Kubernetes"],
        explanation="В целом подходит.",
    )
    assert m.score == 82
    assert m.verdict == "match"


@pytest.mark.parametrize("score", [-1, 101, 150])
def test_match_result_score_out_of_range(score):
    with pytest.raises(ValidationError):
        MatchResult(score=score, verdict="match", explanation="x")


def test_match_result_bad_verdict():
    with pytest.raises(ValidationError):
        MatchResult(score=50, verdict="probably", explanation="x")


# --- Report ---------------------------------------------------------------


def test_report_resume_only():
    rep = Report(
        resume=ResumeFields(category="HR", summary="x", language="ru"),
    )
    assert rep.match is None


def test_report_with_match():
    rep = Report(
        resume=ResumeFields(category="HR", summary="x", language="ru"),
        match=MatchResult(score=10, verdict="no_match", explanation="Не подходит."),
    )
    assert rep.match is not None
    assert rep.match.verdict == "no_match"


# --- Непустые строковые поля (StringConstraints) --------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_resume_fields_blank_category_rejected(blank):
    with pytest.raises(ValidationError):
        ResumeFields(category=blank, summary="x", language="ru")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_resume_fields_blank_summary_rejected(blank):
    with pytest.raises(ValidationError):
        ResumeFields(category="HR", summary=blank, language="ru")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_match_result_blank_explanation_rejected(blank):
    with pytest.raises(ValidationError):
        MatchResult(score=50, verdict="maybe", explanation=blank)


def test_non_blank_str_is_stripped():
    """Валидная строка с краевыми пробелами обрезается, а не отвергается."""
    r = ResumeFields(category="  Python Developer  ", summary="  ok  ", language="en")
    assert r.category == "Python Developer"
    assert r.summary == "ok"


# --- Missing required и граничные значения --------------------------------


def test_resume_fields_missing_required_category():
    with pytest.raises(ValidationError):
        ResumeFields(summary="x", language="ru")  # нет category


def test_resume_fields_missing_required_language():
    with pytest.raises(ValidationError):
        ResumeFields(category="HR", summary="x")  # нет language


@pytest.mark.parametrize("score", [0, 100])
def test_match_result_score_inclusive_bounds(score):
    m = MatchResult(score=score, verdict="maybe", explanation="ok")
    assert m.score == score


def test_match_result_verdict_maybe():
    m = MatchResult(score=50, verdict="maybe", explanation="ok")
    assert m.verdict == "maybe"
