import sys

from app import cli


def test_the_themes_command_runs_every_question_inside_one_event_loop(monkeypatch, capsys):
    """Calling asyncio.run once per question broke the second one: the shared OpenAI client belongs to the first loop."""
    loops = []
    real_run = cli.asyncio.run

    def counting_run(coroutine):
        loops.append(1)
        return real_run(coroutine)

    async def question_numbers(study_id):
        return [1, 2, 3]

    class FakeRun:
        pass

    async def themes_for_question(study_id, number, method, k, save):
        return FakeRun(), 100 + number

    monkeypatch.setattr(cli.asyncio, "run", counting_run)
    monkeypatch.setattr(cli, "_question_numbers", question_numbers)
    monkeypatch.setattr(cli, "themes_for_question", themes_for_question)
    monkeypatch.setattr(cli, "format_run", lambda run: "a run")
    monkeypatch.setattr(sys, "argv", ["cli", "themes", "--study", "9"])

    cli.main()

    assert len(loops) == 1
    assert capsys.readouterr().out.count("saved as theme run") == 3
