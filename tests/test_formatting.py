"""Тесты форматирования ответа бота (RU/EN по языку резюме)."""

from bot.formatting import format_report, format_resume
from service.schemas import MatchResult, Report, ResumeFields


def test_format_resume_ru():
    r = ResumeFields(
        category="Python Developer",
        skills=["Python", "Docker"],
        years_experience=4,
        grade="middle",
        education="МГТУ",
        summary="Бэкенд-разработчик.",
        language="ru",
    )
    out = format_resume(r)
    assert "📋 Разбор резюме" in out
    assert "Категория: Python Developer" in out
    assert "Навыки: Python, Docker" in out
    assert "Грейд: middle" in out


def test_format_resume_en_with_none_optionals():
    r = ResumeFields(category="HR", summary="HR generalist.", language="en")
    out = format_resume(r)
    assert "📋 Resume breakdown" in out
    assert "Grade: —" in out
    assert "Experience (yrs): —" in out
    assert "Skills: —" in out


def test_format_report_resume_only_has_no_match_section():
    r = ResumeFields(category="HR", summary="x", language="ru")
    out = format_report(Report(resume=r))
    assert "🎯" not in out


def test_format_report_with_match_ru():
    r = ResumeFields(category="Python Developer", summary="x", language="ru")
    m = MatchResult(
        score=78, verdict="match", strengths=["Docker"], gaps=["k8s"], explanation="подходит"
    )
    out = format_report(Report(resume=r, match=m))
    assert "🎯 Соответствие вакансии" in out
    assert "Скор: 78/100" in out
    assert "✅ Подходит" in out
    assert "Сильные стороны: Docker" in out
    assert "Пробелы: k8s" in out


def test_format_report_with_match_en_no_match_verdict():
    r = ResumeFields(category="HR", summary="x", language="en")
    m = MatchResult(score=10, verdict="no_match", explanation="not a fit")
    out = format_report(Report(resume=r, match=m))
    assert "❌ No match" in out
    assert "Score: 10/100" in out


def test_format_match_maybe_verdict():
    r = ResumeFields(category="HR", summary="x", language="ru")
    m = MatchResult(score=50, verdict="maybe", explanation="спорно")
    out = format_report(Report(resume=r, match=m))
    assert "🟡 Возможно" in out


def test_format_experience_whole_number_has_no_decimal():
    r = ResumeFields(category="HR", summary="x", language="ru", years_experience=4)
    out = format_resume(r)
    assert "Опыт (лет): 4" in out
    assert "4.0" not in out


def test_format_experience_fractional_kept():
    r = ResumeFields(category="HR", summary="x", language="en", years_experience=4.5)
    out = format_resume(r)
    assert "Experience (yrs): 4.5" in out
