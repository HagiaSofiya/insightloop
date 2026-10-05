from app.prompts.manifest import current_hashes, load_manifest, problems

RECORDED = {"p/v1.md": "aaa", "p/v2.md": "bbb"}


def test_real_prompt_files_match_the_manifest():
    found = problems(load_manifest(), current_hashes())

    assert found == [], "\n".join(found)


def test_unchanged_files_pass():
    assert problems(RECORDED, dict(RECORDED)) == []


def test_edited_file_without_a_new_version_fails():
    current = {**RECORDED, "p/v1.md": "changed"}

    found = problems(RECORDED, current)

    assert len(found) == 1
    assert "p/v1.md changed but its version did not" in found[0]


def test_new_version_must_be_recorded():
    found = problems(RECORDED, {**RECORDED, "p/v3.md": "ccc"})

    assert len(found) == 1
    assert "p/v3.md is not in the manifest" in found[0]


def test_deleted_version_fails():
    found = problems(RECORDED, {"p/v1.md": "aaa"})

    assert len(found) == 1
    assert "p/v2.md was deleted" in found[0]


def test_hashes_follow_file_content(tmp_path):
    (tmp_path / "p").mkdir()
    file = tmp_path / "p" / "v1.md"
    file.write_text("hello")
    before = current_hashes(tmp_path)

    file.write_text("hello!")

    assert current_hashes(tmp_path) != before
    assert list(before) == ["p/v1.md"]
