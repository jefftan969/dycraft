"""A task too long for one argument reaches the agent as a file.

Linux refuses to start a program any one of whose arguments or environment
strings is over 128 KiB (`MAX_ARG_STRLEN`), with E2BIG and nothing from the
program. The launcher hands the agent CLI its task inside the first message,
which Codex receives together with the system prompt as one argument, and
exports it to the daemon as `$DYCRAFT_USER_PROMPT`. A review's prompt embeds
the plan and its evidence, and Linux refuses to start a program any one of
whose arguments is 128 KiB or more.
"""
from dycraft.agent.agent_runner import (INLINE_TASK_LIMIT, TASK_FILE, TASK_FILE_ENV,
                                        compose_footer, compose_system_prompt,
                                        stage_long_task, user_prompt_from_env)

MAX_ARG_STRLEN = 131_072


def _footer(layer, run_dir):
    return lambda task: compose_footer(layer, task, run_dir)


def test_a_task_that_fits_is_passed_inline(tmp_path):
    system = compose_system_prompt("animation")
    task, path = stage_long_task(tmp_path, "A robot waters the plants.", system,
                                 _footer("animation", tmp_path))
    assert task == "A robot waters the plants." and path is None
    assert not (tmp_path / TASK_FILE).exists()


def test_a_task_too_long_for_one_argument_goes_to_a_file(tmp_path):
    system = compose_system_prompt("plain")
    review = "Review this film.\n" + '{"plan": "x"}\n' * (16 * 1024)
    assert len(review.encode()) > MAX_ARG_STRLEN
    footer = _footer("plain", tmp_path)
    task, path = stage_long_task(tmp_path, review, system, footer)
    assert path == tmp_path / TASK_FILE and path.read_text() == review
    assert str(path) in task and "Read all of it" in task
    # What the CLI is then handed fits, with room for what it adds.
    assert len((system + "\n\n" + footer(task)).encode()) < INLINE_TASK_LIMIT


def test_the_limit_counts_the_system_prompt_codex_receives_with_the_task(tmp_path):
    """A task well under 128 KiB still fails on Codex when the system prompt
    beside it takes the rest of the argument, so the same task that passes
    inline on its own is staged beside the real system prompt."""
    system = compose_system_prompt("scene")
    footer = _footer("scene", tmp_path)
    task = "x" * (INLINE_TASK_LIMIT - len(footer("").encode()) - len(system.encode()) // 2)
    assert stage_long_task(tmp_path, task, "", footer) == (task, None)
    staged, path = stage_long_task(tmp_path, task, system, footer)
    assert path == tmp_path / TASK_FILE and path.read_text() == task
    assert str(path) in staged and task not in staged


def test_the_daemon_reads_the_task_from_the_file(tmp_path):
    (tmp_path / TASK_FILE).write_text("the whole review prompt")
    env = {TASK_FILE_ENV: str(tmp_path / TASK_FILE), "DYCRAFT_USER_PROMPT": "Your task is in ..."}
    assert user_prompt_from_env(env) == "the whole review prompt"
    assert user_prompt_from_env({"DYCRAFT_USER_PROMPT": "inline"}) == "inline"


def test_a_child_does_not_inherit_its_parents_task_file(tmp_path, monkeypatch):
    from dycraft.cli.cli import build_env_dict

    monkeypatch.setenv(TASK_FILE_ENV, "/parent/run/task.md")
    env = build_env_dict(
        "the child's own brief", run_dir=tmp_path, runs_dir=tmp_path, run_id="r",
        layer="object", runtime="codex_cli")
    assert TASK_FILE_ENV not in env
    assert user_prompt_from_env(env) == "the child's own brief"
