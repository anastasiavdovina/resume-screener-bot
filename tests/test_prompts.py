"""Тесты загрузки YAML-промптов и рендера шаблонов."""

import pytest

from llm.prompt_loader import Prompt, PromptSet, default_prompt_set, load_prompts


def test_load_default_prompts():
    ps = load_prompts()
    assert isinstance(ps, PromptSet)
    assert ps.parse_resume.system.strip()
    assert ps.score_match.system.strip()
    assert "{resume}" in ps.parse_resume.user_template
    assert "{resume}" in ps.score_match.user_template
    assert "{vacancy}" in ps.score_match.user_template


def test_prompt_metadata_loaded():
    ps = load_prompts()
    assert ps.parse_resume.name == "parse_resume"
    assert ps.parse_resume.version == 1


def test_default_prompt_set_is_cached():
    assert default_prompt_set() is default_prompt_set()


def test_prompt_render_substitutes():
    p = Prompt(name="t", version=1, system="s", user_template="A {x} B {y}")
    assert p.render(x="1", y="2") == "A 1 B 2"


def test_prompt_render_no_cross_placeholder_collision():
    # {b}, попавший в значение для {a}, НЕ должен замениться значением b
    p = Prompt(name="t", version=1, system="s", user_template="{a}|{b}")
    assert p.render(a="{b}", b="Z") == "{b}|Z"


def test_prompt_render_unknown_placeholder_kept():
    p = Prompt(name="t", version=1, system="s", user_template="hello {missing}")
    assert p.render() == "hello {missing}"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path)  # пустой каталог


def test_load_missing_key_raises(tmp_path):
    (tmp_path / "parse_resume.yaml").write_text(
        "name: x\nversion: 1\nsystem: s\n", encoding="utf-8"  # нет user_template
    )
    with pytest.raises(ValueError):
        load_prompts(tmp_path)
