"""Сборка человекочитаемого ответа из `Report`.

Результат анализа выводится на языке резюме (`report.resume.language`). Текст —
без markdown, чтобы спецсимволы из резюме/вакансии не ломали разметку Telegram.
"""

from __future__ import annotations

from service.schemas import MatchResult, Report, ResumeFields

_LABELS = {
    "ru": {
        "resume_title": "📋 Разбор резюме",
        "category": "Категория",
        "grade": "Грейд",
        "experience": "Опыт (лет)",
        "skills": "Навыки",
        "education": "Образование",
        "summary": "Кратко",
        "match_title": "🎯 Соответствие вакансии",
        "score": "Скор",
        "verdict": "Вердикт",
        "strengths": "Сильные стороны",
        "gaps": "Пробелы",
        "explanation": "Обоснование",
        "none": "—",
    },
    "en": {
        "resume_title": "📋 Resume breakdown",
        "category": "Category",
        "grade": "Grade",
        "experience": "Experience (yrs)",
        "skills": "Skills",
        "education": "Education",
        "summary": "Summary",
        "match_title": "🎯 Vacancy match",
        "score": "Score",
        "verdict": "Verdict",
        "strengths": "Strengths",
        "gaps": "Gaps",
        "explanation": "Explanation",
        "none": "—",
    },
}

_VERDICT = {
    "ru": {"match": "✅ Подходит", "maybe": "🟡 Возможно", "no_match": "❌ Не подходит"},
    "en": {"match": "✅ Match", "maybe": "🟡 Maybe", "no_match": "❌ No match"},
}


def format_resume(resume: ResumeFields) -> str:
    labels = _LABELS[resume.language]
    none = labels["none"]
    # :g убирает хвостовой .0 у целых (4.0 -> "4"), сохраняя дробные (4.5 -> "4.5")
    experience = none if resume.years_experience is None else f"{resume.years_experience:g}"
    return "\n".join(
        [
            labels["resume_title"],
            f"{labels['category']}: {resume.category}",
            f"{labels['grade']}: {resume.grade or none}",
            f"{labels['experience']}: {experience}",
            f"{labels['skills']}: {', '.join(resume.skills) if resume.skills else none}",
            f"{labels['education']}: {resume.education or none}",
            f"{labels['summary']}: {resume.summary}",
        ]
    )


def format_match(match: MatchResult, language: str) -> str:
    labels = _LABELS[language]
    lines = [
        labels["match_title"],
        f"{labels['score']}: {match.score}/100",
        f"{labels['verdict']}: {_VERDICT[language][match.verdict]}",
    ]
    if match.strengths:
        lines.append(f"{labels['strengths']}: " + "; ".join(match.strengths))
    if match.gaps:
        lines.append(f"{labels['gaps']}: " + "; ".join(match.gaps))
    lines.append(f"{labels['explanation']}: {match.explanation}")
    return "\n".join(lines)


def format_report(report: Report) -> str:
    """Собрать полный ответ: разбор резюме и (если есть) оценку соответствия."""
    language = report.resume.language
    parts = [format_resume(report.resume)]
    if report.match is not None:
        parts.append(format_match(report.match, language))
    return "\n\n".join(parts)
