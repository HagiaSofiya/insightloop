import pytest

from app.prompts import registry


@pytest.fixture
def prompts_dir(tmp_path, monkeypatch):
    (tmp_path / "greet").mkdir()
    (tmp_path / "greet" / "v1.md").write_text("Hello $name, welcome to $place.")
    monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
    return tmp_path


def test_variables_are_filled_in(prompts_dir):
    assert registry.load("greet", "v1", name="Sam", place="Insightloop") == (
        "Hello Sam, welcome to Insightloop."
    )


def test_missing_variable_raises(prompts_dir):
    with pytest.raises(KeyError):
        registry.load("greet", "v1", name="Sam")


def test_unknown_prompt_or_version_raises(prompts_dir):
    with pytest.raises(FileNotFoundError):
        registry.load("greet", "v2", name="Sam", place="x")


def test_real_guide_prompt_loads_without_variables():
    assert "discussion guide" in registry.load("discussion_guide", "v1")
