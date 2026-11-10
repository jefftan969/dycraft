"""The probe that decides whether a Codex agent may start: Codex's sandbox must start and confine writes.

The fakes stand in for `codex sandbox`, so each outcome is reproduced without the real CLI; the last test runs the
real one where it is installed (no model call).
"""
import json
import os
import shutil
from pathlib import Path

import pytest

from dycraft.cli import backends as B


@pytest.fixture(autouse=True)
def codex_logged_in(monkeypatch):
    """The account's login, which `check_codex` and the preflight read from its home, taken as present: these tests
    are about the sandbox."""
    monkeypatch.setattr(B.codex, "codex_account_problem", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_namespace(monkeypatch):
    """No `unshare` wrapper unless a test gives one: whether the real one works depends on the host."""
    monkeypatch.setattr(B.codex, "codex_namespace_wrapper", lambda: [])


def fake_wrapper(tmp_path) -> list[str]:
    """A stand-in for `codex_namespace_wrapper`: runs its command with NAMESPACE=1."""
    path = tmp_path / "wrap"
    path.write_text('#!/bin/sh\nNAMESPACE=1 exec "$@"\n')
    path.chmod(0o755)
    return [str(path)]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Where the fakes' outside writes go. The real choice avoids `/tmp`, which holds `tmp_path`, and has its
    own test below."""
    monkeypatch.setattr(B.codex, "_probe_outside_parent", lambda run_dir, env: tmp_path / "cache" / "dycraft")
    return tmp_path / "cache"


def fake(tmp_path, body: str) -> str:
    """A `codex` whose `sandbox` subcommand runs `body` in sh, with the probe's inside and outside paths as
    $INSIDE and $OUTSIDE."""
    path = tmp_path / "codex"
    path.write_text("#!/bin/sh\n"
                    'for a in "$@"; do PREV2="$PREV"; PREV="$a"; done\n'
                    'INSIDE="$PREV2"; OUTSIDE="$PREV"\n' + body + "\n")
    path.chmod(0o755)
    return str(path)


def run_dir(tmp_path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def test_a_confining_sandbox_passes_and_leaves_nothing_behind(tmp_path, cache):
    codex = fake(tmp_path, 'echo ok > "$INSIDE"; echo CONFINED')
    run = run_dir(tmp_path)
    assert B.codex_sandbox_probe(codex, str(tmp_path / "home"), run) is None
    assert list(run.iterdir()) == []
    assert list((cache / "dycraft").iterdir()) == []


def test_a_sandbox_that_cannot_start_is_reported_with_its_error(tmp_path, cache):
    codex = fake(tmp_path, "echo \"bwrap: Can't mkdir /tmp/.git: No space left on device\" >&2; exit 1")
    problem = B.codex_sandbox_probe(codex, str(tmp_path / "home"), run_dir(tmp_path))
    assert problem == "the sandbox failed to start: bwrap: Can't mkdir /tmp/.git: No space left on device"


def test_the_reported_error_comes_from_stderr_never_from_the_probe_s_own_output(tmp_path, cache):
    codex = fake(tmp_path, 'echo CONFINED; exit 1')
    assert B.codex_sandbox_probe(codex, str(tmp_path / "home"), run_dir(tmp_path)) == \
        "the sandbox failed to start: exit status 1"


def test_a_refused_launch_is_recorded_for_the_parent(tmp_path, cache, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    """A refused child leaves `meta.json` saying why, so its parent does not see an unexplained dead launch."""
    monkeypatch.delenv(B.ALLOW_UNSANDBOXED_CODEX_ENV, raising=False)
    monkeypatch.setattr(B.codex, "codex_sandbox_probe", lambda *a, **k: "the sandbox failed to start: bwrap: x")
    monkeypatch.setattr(B.CodexBackend, "_resolve_codex_cli",
                        classmethod(lambda cls: (fake(tmp_path, "true"), str(tmp_path / "home"))))
    run = run_dir(tmp_path)
    assert B.CodexBackend().preflight(run, "animation") == 2
    meta = json.loads((run / "meta.json").read_text())
    failure = meta["launch_failure"]
    assert failure["reason"] == "codex_sandbox_refused"
    # Every setup tried is named with what it reported, the inherited one first and a /tmp of its own last.
    monkeypatch.setattr(B.codex, "codex_namespace_wrapper", lambda: ["/bin/unshare"])
    run2 = tmp_path / "run2"
    run2.mkdir()
    assert B.CodexBackend().preflight(run2, "animation") == 2
    detail = json.loads((run2 / "meta.json").read_text())["launch_failure"]["detail"]
    assert detail.startswith("it failed in every setup tried: TMPDIR=")
    assert detail.count("the sandbox failed to start: bwrap: x") == 5
    assert f"; a /tmp of its own (unshare -r --mount, TMPDIR={run2.resolve() / '.tmp'}): the sandbox" in detail
    assert failure["detail"].count("the sandbox failed to start: bwrap: x") == 4
    # The whole opening sentence on one line, and what to do, for the run's README.
    assert failure["message"] == ("FATAL: Codex's sandbox does not work here, so DyCraft will not start a Codex agent "
                                  "without write confinement.")
    assert failure["fix"] == B.codex.codex_sandbox_fixes()


def test_a_sandbox_that_does_not_confine_is_caught(tmp_path, cache):
    codex = fake(tmp_path, 'echo ok > "$INSIDE"; echo x > "$OUTSIDE"; echo CONFINED')
    run = run_dir(tmp_path)
    assert B.codex_sandbox_probe(codex, str(tmp_path / "home"), run) == \
        "a write outside the run directory succeeded"
    assert list((cache / "dycraft").iterdir()) == []


def test_a_hung_probe_times_out(tmp_path, cache):
    codex = fake(tmp_path, "sleep 30")
    assert B.codex_sandbox_probe(codex, str(tmp_path / "home"), run_dir(tmp_path), timeout=1) == \
        "the probe timed out after 1 s"


def test_the_probe_runs_with_the_launch_account(tmp_path, cache):
    codex = fake(tmp_path, 'echo ok > "$INSIDE"; echo "$CODEX_HOME" > "$INSIDE"; echo CONFINED; '
                           'cp "$INSIDE" "$INSIDE.home"')
    run = run_dir(tmp_path)
    B.codex_sandbox_probe(codex, "/accounts/codex3", run)
    homes = list(run.glob("*.home"))
    assert homes and homes[0].read_text().strip() == "/accounts/codex3"


def test_the_outside_write_is_never_under_a_writable_root(tmp_path, monkeypatch):
    """`/tmp` and `$TMPDIR` are writable inside the sandbox, so a write there proves nothing."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    monkeypatch.setenv("HOME", "/tmp/home")
    run = Path("/srv/runs/r1")
    assert B.codex._probe_outside_parent(run, {}) == Path("/srv/runs")
    assert B.codex._probe_outside_parent(Path("/tmp/runs/r1"), {}) is None
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert B.codex._probe_outside_parent(run, {"TMPDIR": str(tmp_path)}) == Path("/srv/runs")


def test_a_scratch_directory_that_cannot_be_made_is_reported_not_raised(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(B.codex, "_probe_outside_parent", lambda run_dir, env: blocker / "dycraft")
    problem = B.codex_sandbox_probe(fake(tmp_path, "echo CONFINED"), str(tmp_path), run_dir(tmp_path))
    assert problem.startswith("the probe could not make a scratch directory")


def test_the_refusal_names_the_cause_the_fixes_and_the_opt_in(apparmor):
    apparmor(restricted=True, bwrap=False, profiled=True)          # Ubuntu with a bwrap profile: the package does it
    text = B.codex_sandbox_refusal("the probe timed out after 60 s", "/bin/codex", "/home/u/.codex",
                                   "codex-cli 0.153.4")
    assert text.startswith("FATAL: Codex's sandbox does not work here, so DyCraft will not start a Codex agent "
                           "without write confinement.\n")      # the approved text (writing pass B3.3, 11-09)
    assert "result: the probe timed out after 60 s" in text
    assert "sudo apt-get install bubblewrap" in text and "--backend claude" in text
    assert "pass --allow-unsandboxed to run Codex without its sandbox" in text


@pytest.mark.skipif(not shutil.which(os.environ.get("DYCRAFT_CODEX_EXECUTABLE") or "codex"),
                    reason="the Codex CLI is not installed")
def test_the_real_codex_sandbox_confines_writes_here():
    """On a machine where Codex's sandbox works, the probe passes; anywhere else it names the failure."""
    binary = shutil.which(os.environ.get("DYCRAFT_CODEX_EXECUTABLE") or "codex")
    run = Path.home() / ".cache" / "dycraft" / f"probe_test_{os.getpid()}"
    run.mkdir(parents=True, exist_ok=True)
    try:
        problem = B.codex_sandbox_probe(binary, str(Path.home() / ".codex"), run)
    finally:
        shutil.rmtree(run, ignore_errors=True)
    assert problem is None or problem.startswith(("the sandbox failed to start", "the probe"))


def _bundle(home, contents):
    home.mkdir(parents=True, exist_ok=True)
    (home / "cloud-config-bundle-cache.json").write_text(json.dumps({"signed_payload": {"bundle": {
        "requirements_toml": {"enterprise_managed": [{"id": "x", "contents": contents}]}}}}))


def test_an_organizations_approval_restriction_is_read_from_the_account(tmp_path):
    _bundle(tmp_path / "org", 'allowed_approval_policies = [ "on-request", "untrusted" ]\n')
    assert B.codex_allowed_approval_policies(str(tmp_path / "org")) == ["on-request", "untrusted"]
    _bundle(tmp_path / "free", '[features]\nchronicle = false\n')
    assert B.codex_allowed_approval_policies(str(tmp_path / "free")) is None
    assert B.codex_allowed_approval_policies(str(tmp_path / "none")) is None


def test_preflight_warns_and_records_when_the_organization_forbids_never(tmp_path, monkeypatch, capsys):
    """`never` cannot be enforced there: Codex falls back to on-request with its reviewer, and the run says so."""
    home = tmp_path / "org"
    _bundle(home, 'allowed_approval_policies = [ "on-request", "untrusted" ]\n')
    monkeypatch.setattr(B.codex, "codex_sandbox_probe", lambda *a, **k: None)
    monkeypatch.setattr(B.CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: ("/bin/codex", str(home))))
    backend = B.CodexBackend()
    assert backend.preflight(tmp_path) is None
    assert backend.launch_meta["codex_sandbox"]["approval"] == "on-request (organization policy)"
    assert "does not allow `--ask-for-approval never`" in capsys.readouterr().err


def test_the_session_s_own_approval_policy_is_recorded_and_an_override_is_reported(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"codex_sandbox": {"probe": "ok", "approval": "never"}}))
    watch = B._process._CodexRolloutWatch(["codex", "-C", str(run), "--ask-for-approval", "never"], {"CODEX_HOME": "/x"})
    watch._note_approval_policy({"approval_policy": "on-request"})
    meta = json.loads((run / "meta.json").read_text())
    assert meta["codex_sandbox"] == {"probe": "ok", "approval": "never", "approval_effective": "on-request"}
    assert "not `never`" in capsys.readouterr().err
    watch = B._process._CodexRolloutWatch(["codex", "-C", str(run), "--ask-for-approval", "never"], {"CODEX_HOME": "/x"})
    watch._note_approval_policy({"approval_policy": "never"})
    assert json.loads((run / "meta.json").read_text())["codex_sandbox"]["approval_effective"] == "never"
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# A TMPDIR the sandbox cannot use: a cluster job's own (Slurm's /scr/<job>/)
# ---------------------------------------------------------------------------

_SCR = "the sandbox failed to start: bwrap: Can't mkdir parents for /scr/1/.1/_tmp/codex-daemon-1: Permission denied"


def _probe_tmpdir_matters(run_dir):
    """A probe that fails unless TMPDIR is the run's own, as on an SC compute node."""
    def probe(binary, home, run, env=None, wrapper=()):
        return None if (env or {}).get("TMPDIR") == str(Path(run_dir).resolve() / ".tmp") else _SCR
    return probe


def test_a_tmpdir_the_sandbox_cannot_use_is_replaced_by_the_runs_own(tmp_path, monkeypatch):
    from dycraft.cli.backends import codex as X
    monkeypatch.setattr(X, "codex_sandbox_probe", _probe_tmpdir_matters(tmp_path))
    found = X.codex_sandbox_probe_or_tmpdir("codex", "/h", tmp_path, {"TMPDIR": "/scr/1"})
    assert found == (None, str(tmp_path.resolve() / ".tmp"), _SCR, [])
    assert (tmp_path / ".tmp").is_dir()


def test_a_sandbox_that_fails_every_way_names_each_tmpdir_tried(tmp_path, monkeypatch):
    from dycraft.cli.backends import codex as X
    monkeypatch.setattr(X, "codex_sandbox_probe", lambda *a, **k: "the sandbox failed to start: bwrap: no")
    env = {"TMPDIR": "/scr/1", "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    problem, tmpdir, inherited, wrapper = X.codex_sandbox_probe_or_tmpdir("codex", "/h", tmp_path, env)
    assert (tmpdir, inherited, wrapper) == (None, None, [])
    assert problem.startswith("it failed in every setup tried: TMPDIR=/scr/1: the sandbox failed to start")
    for fallback in X._fallback_tmpdirs(tmp_path, env):
        assert f"TMPDIR={fallback}: the sandbox failed to start: bwrap: no" in problem


# The real error from an SC GPU node, reproduced: a `codex sandbox` whose bwrap cannot make the mount point for a
# directory under the TMPDIR it was given. Paths under `bad` play the job's /scr.

def _scr_codex(tmp_path, *bad: Path) -> str:
    cases = "|".join(f'"{p}"*' for p in bad)
    return fake(tmp_path, f'case "$TMPDIR" in {cases}) '
                          "echo \"bwrap: Can't mkdir parents for $TMPDIR/codex-daemon-$$: Permission denied\" >&2; "
                          'exit 1;; esac; echo ok > "$INSIDE"; echo CONFINED')


def test_the_sc_error_is_retried_in_the_runs_own_tmpdir(tmp_path, cache):
    from dycraft.cli.backends import codex as X
    scr = tmp_path / "scr" / "18402613" / ".18402613" / "_tmp"
    scr.mkdir(parents=True)
    run = run_dir(tmp_path)
    problem, tmpdir, inherited, _ = X.codex_sandbox_probe_or_tmpdir(_scr_codex(tmp_path, scr), str(tmp_path), run,
                                                                    {**os.environ, "TMPDIR": str(scr)})
    assert (problem, tmpdir) == (None, str(run.resolve() / ".tmp"))
    assert inherited.startswith(f"the sandbox failed to start: bwrap: Can't mkdir parents for {scr}/codex-daemon-")
    assert inherited.endswith(": Permission denied")


def test_when_the_run_directory_will_not_do_either_the_next_tmpdir_is_tried(tmp_path, cache):
    from dycraft.cli.backends import codex as X
    scr = tmp_path / "scr"
    scr.mkdir()
    run = run_dir(tmp_path)
    env = {**os.environ, "TMPDIR": str(scr)}
    problem, tmpdir, _, _ = X.codex_sandbox_probe_or_tmpdir(_scr_codex(tmp_path, scr, run.resolve()), str(tmp_path),
                                                            run, env)
    assert problem is None and tmpdir == str(X._fallback_tmpdirs(run, env)[1])


def test_when_no_tmpdir_works_each_one_tried_is_named_with_its_own_error(tmp_path, cache, monkeypatch):
    from dycraft.cli.backends import codex as X
    scr = tmp_path / "scr"
    scr.mkdir()
    run = run_dir(tmp_path)
    env = {**os.environ, "TMPDIR": str(scr), "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    fallbacks = X._fallback_tmpdirs(run, env)
    codex = _scr_codex(tmp_path, scr, *(f.resolve() for f in fallbacks))
    problem, tmpdir, inherited, _ = X.codex_sandbox_probe_or_tmpdir(codex, str(tmp_path), run, env)
    assert (tmpdir, inherited) == (None, None)
    assert problem.startswith("it failed in every setup tried: ")
    for where in (scr, *fallbacks):
        assert f"TMPDIR={where}: the sandbox failed to start: bwrap: Can't mkdir parents for {where}" in problem


def test_a_write_that_leaked_is_not_retried(tmp_path, monkeypatch):
    from dycraft.cli.backends import codex as X
    calls = []
    monkeypatch.setattr(X, "codex_sandbox_probe",
                        lambda *a, **k: calls.append(1) or "a write outside the run directory succeeded")
    assert X.codex_sandbox_probe_or_tmpdir("codex", "/h", tmp_path)[0].startswith("a write outside")
    assert calls == [1]


def test_a_launch_runs_codex_with_the_runs_tmpdir_and_says_so(tmp_path, monkeypatch, capsys):
    from dycraft.cli.backends import codex as X
    monkeypatch.setenv("TMPDIR", "/scr/1")
    monkeypatch.delenv("DYCRAFT_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(X, "codex_account_problem", lambda *a, **k: None)
    monkeypatch.setattr(X, "codex_sandbox_probe", _probe_tmpdir_matters(tmp_path))
    monkeypatch.setattr(X.CodexBackend, "_resolve_codex_cli", lambda self: ("/bin/codex", str(tmp_path / "home")))
    monkeypatch.setattr(X, "codex_allowed_approval_policies", lambda home: None)
    backend = X.CodexBackend()
    assert backend.preflight(tmp_path, "object") is None
    assert f"[cli] Codex's sandbox cannot use TMPDIR=/scr/1 ({_SCR}); Codex runs with TMPDIR=" \
           f"{tmp_path.resolve() / '.tmp'} instead" in capsys.readouterr().err
    assert backend.launch_meta["codex_sandbox"]["tmpdir"] == str(tmp_path.resolve() / ".tmp")
    seen = {}
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setattr(X._process, "run_agent_cli", lambda cmd, env, watch=None: seen.update(env=env) or 0)
    monkeypatch.setattr(X, "mark_codex_untrusted", lambda *a, **k: (lambda: None))
    from dycraft.cli.backends import LaunchPlan
    backend.execute(LaunchPlan(layer="object", user_prompt="p", run_dir=tmp_path, runs_dir=tmp_path.parent,
                               run_id="r", model=None, env={"TMPDIR": "/scr/1"}, system_prompt_text="s",
                               footer_text="f"))
    assert seen["env"]["TMPDIR"] == str(tmp_path.resolve() / ".tmp")


def test_the_check_says_launches_use_the_runs_tmpdir(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.cli.backends import CodexBackend
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\necho codex-cli 0.153.4\n")
    fake.chmod(0o755)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(tmp_path))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: (None, "/run/.tmp", _SCR, []))
    monkeypatch.setenv("TMPDIR", "/scr/1")
    ctx = SimpleNamespace(facts={})
    result = E.check_codex(ctx)
    assert result.status == E.WARN
    assert result.summary.endswith("its sandbox cannot use TMPDIR=/scr/1, so each launch runs Codex with TMPDIR "
                                   "inside its run directory")
    assert any(_SCR in line for line in result.detail)
    assert ctx.facts["codex_launch"] == {"wrapper": [], "tmpdir": ".tmp"}


def test_the_check_says_launches_use_a_namespace_and_its_probes_do_too(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.cli.backends import CodexBackend
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\necho codex-cli 0.167.0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(tmp_path))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: (None, "/run/.tmp", _SCR, ["/w", "-r"]))
    ctx = SimpleNamespace(facts={})
    result = E.check_codex(ctx)
    assert result.status == E.WARN
    assert result.summary.endswith("its sandbox cannot start with this machine's /tmp, so each launch runs Codex in "
                                   "a namespace of its own with a fresh /tmp, and TMPDIR inside its run directory")
    # The capability probes start the agentic Codex the same way, and only that one.
    probe = E.HarnessProbe.__new__(E.HarnessProbe)
    probe.ctx, probe.spec, probe.outputs, probe.model, probe.effort = ctx, E.builtin_spec("codex"), [], "", ""
    seen = {}
    monkeypatch.setattr(E, "run", lambda cmd, **kw: seen.update(cmd=cmd, env=kw["env"]) or E.Ran(0, "", ""))
    probe.launch(probe.spec.agentic, prompt="p", cwd=tmp_path / "w", env={"PATH": "/bin"})
    assert seen["cmd"][:2] == ["/w", "-r"] and seen["env"]["TMPDIR"] == str(tmp_path / "w" / ".tmp")
    probe.launch(probe.spec.oneshot, prompt="p", cwd=tmp_path / "w", env={"PATH": "/bin"})
    assert seen["cmd"][0] != "/w" and "TMPDIR" not in seen["env"]


def test_the_check_reports_codex_runs_only_with_the_opt_in(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dycraft import environment_test as E
    from dycraft.cli.backends import CodexBackend
    fake = tmp_path / "codex"
    fake.write_text("#!/bin/sh\necho codex-cli 0.167.0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(CodexBackend, "_resolve_codex_cli", classmethod(lambda cls: (str(fake), str(tmp_path))))
    monkeypatch.setattr(E, "_codex_sandbox_problem", lambda binary, home: ("the sandbox failed to start: x", None,
                                                                           None, []))
    result = E.check_codex(SimpleNamespace(facts={}))
    assert result.status == E.DEGRADED and result.data == {"sandbox": True}
    assert "so Codex runs only with --allow-unsandboxed (unconfined): the sandbox failed to start: x" in result.summary
    assert any(line.startswith("without sudo: `--allow-unsandboxed`") for line in result.fix)


# ---------------------------------------------------------------------------
# A /tmp the sandbox cannot use at all: Codex in a namespace of its own
# ---------------------------------------------------------------------------

def _namespace_codex(tmp_path, *, leak=False) -> str:
    """A `codex sandbox` that starts only in the namespace (NAMESPACE=1), whatever TMPDIR it is given."""
    write = 'echo x > "$OUTSIDE"; ' if leak else ""
    return fake(tmp_path, '[ "$NAMESPACE" = 1 ] || { echo "bwrap: Can\'t mkdir parents for /scr/1/.1/_tmp/'
                          'codex-daemon-1: Permission denied" >&2; exit 1; }; '
                          f'echo ok > "$INSIDE"; {write}echo CONFINED; echo "$TMPDIR" > "$INSIDE.tmpdir"')


def test_when_no_tmpdir_works_codex_gets_a_namespace_with_a_tmp_of_its_own(tmp_path, cache, monkeypatch):
    from dycraft.cli.backends import codex as X
    wrapper = fake_wrapper(tmp_path)
    monkeypatch.setattr(X, "codex_namespace_wrapper", lambda: wrapper)
    run = run_dir(tmp_path)
    env = {**os.environ, "TMPDIR": "/scr/1", "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    problem, tmpdir, inherited, used = X.codex_sandbox_probe_or_tmpdir(_namespace_codex(tmp_path), str(tmp_path),
                                                                       run, env)
    assert (problem, tmpdir, used) == (None, str(run.resolve() / ".tmp"), wrapper)
    assert inherited.startswith("the sandbox failed to start: bwrap: Can't mkdir parents for /scr/1/")
    # The probe ran with the run's own TMPDIR inside the namespace.
    assert [p.read_text().strip() for p in run.glob("*.tmpdir")] == [str(run.resolve() / ".tmp")]


def test_a_namespace_whose_probe_leaks_is_not_used(tmp_path, cache, monkeypatch):
    from dycraft.cli.backends import codex as X
    wrapper = fake_wrapper(tmp_path)
    monkeypatch.setattr(X, "codex_namespace_wrapper", lambda: wrapper)
    env = {**os.environ, "TMPDIR": "/scr/1", "XDG_CACHE_HOME": str(tmp_path / "xdg")}
    found = X.codex_sandbox_probe_or_tmpdir(_namespace_codex(tmp_path, leak=True), str(tmp_path), run_dir(tmp_path),
                                            env)
    assert found == ("a write outside the run directory succeeded", None, None, [])


def test_the_wrapper_gives_codex_a_fresh_tmp_and_is_absent_without_unshare(monkeypatch):
    from dycraft.cli.backends import codex as X
    monkeypatch.undo()
    monkeypatch.setattr(X.shutil, "which", lambda name: "/usr/bin/unshare" if name == "unshare" else None)
    wrapper = X.codex_namespace_wrapper()
    assert wrapper[:3] == ["/usr/bin/unshare", "-r", "--mount"]
    assert f"mount -t tmpfs -o size={X.CODEX_TMPFS_SIZE},mode=1777 tmpfs /tmp && exec \"$@\"" in wrapper
    monkeypatch.setattr(X.shutil, "which", lambda name: None)
    assert X.codex_namespace_wrapper() == []


def test_a_launch_runs_codex_in_the_namespace_and_says_so(tmp_path, monkeypatch, capsys):
    from dycraft.cli.backends import codex as X
    monkeypatch.setenv("TMPDIR", "/scr/1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("DYCRAFT_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(X, "codex_account_problem", lambda *a, **k: None)
    monkeypatch.setattr(X, "codex_sandbox_probe", lambda binary, home, run, env=None, wrapper=(): None
                        if wrapper == ["/w/unshare", "-r"] else _SCR)
    monkeypatch.setattr(X, "codex_namespace_wrapper", lambda: ["/w/unshare", "-r"])
    monkeypatch.setattr(X.CodexBackend, "_resolve_codex_cli", lambda self: ("/bin/codex", str(tmp_path / "home")))
    monkeypatch.setattr(X, "codex_allowed_approval_policies", lambda home: None)
    backend = X.CodexBackend()
    assert backend.preflight(tmp_path, "object") is None
    assert (f"[cli] Codex's sandbox cannot start with this machine's /tmp ({_SCR}); Codex runs in a namespace of its "
            f"own with a fresh /tmp, and TMPDIR={tmp_path.resolve() / '.tmp'}") in capsys.readouterr().err
    record = backend.launch_meta["codex_sandbox"]
    assert record["namespace"] == f"its own /tmp (tmpfs, {X.CODEX_TMPFS_SIZE})"
    assert record["tmpdir"] == str(tmp_path.resolve() / ".tmp")
    seen = {}
    monkeypatch.setenv("DYCRAFT_CODEX_NO_PTY_WRAP", "1")
    monkeypatch.setattr(X._process, "run_agent_cli", lambda cmd, env, watch=None: seen.update(cmd=cmd, env=env) or 0)
    monkeypatch.setattr(X, "mark_codex_untrusted", lambda *a, **k: (lambda: None))
    from dycraft.cli.backends import LaunchPlan
    backend.execute(LaunchPlan(layer="object", user_prompt="p", run_dir=tmp_path, runs_dir=tmp_path.parent,
                               run_id="r", model=None, env={"TMPDIR": "/scr/1"}, system_prompt_text="s",
                               footer_text="f"))
    assert seen["cmd"][:3] == ["/w/unshare", "-r", "/bin/codex"]
    assert seen["env"]["TMPDIR"] == str(tmp_path.resolve() / ".tmp")


def test_the_namespace_can_be_forced_on_or_off_for_diagnosis(tmp_path, cache, monkeypatch):
    from dycraft.cli.backends import codex as X
    wrapper = fake_wrapper(tmp_path)
    monkeypatch.setattr(X, "codex_namespace_wrapper", lambda: wrapper)
    run = run_dir(tmp_path)
    works_anywhere = fake(tmp_path, 'echo ok > "$INSIDE"; echo "$NAMESPACE" > "$INSIDE.ns"; echo CONFINED')
    env = {**os.environ, "DYCRAFT_CODEX_NAMESPACE": "always"}
    problem, tmpdir, why, used = X.codex_sandbox_probe_or_tmpdir(works_anywhere, str(tmp_path), run, env)
    assert (problem, tmpdir, why, used) == (None, str(run.resolve() / ".tmp"), "DYCRAFT_CODEX_NAMESPACE=always",
                                            wrapper)
    assert [p.read_text().strip() for p in run.glob("*.ns")] == ["1"]
    # `never`: the setups without it are tried, and nothing else.
    env = {**os.environ, "DYCRAFT_CODEX_NAMESPACE": "never", "TMPDIR": "/scr/1", "XDG_CACHE_HOME": str(tmp_path / "x")}
    problem, *_ = X.codex_sandbox_probe_or_tmpdir(_namespace_codex(tmp_path), str(tmp_path), run, env)
    assert problem.startswith("it failed in every setup tried: ") and "a /tmp of its own" not in problem


def test_the_capability_probes_keep_their_scratch_off_tmp_when_codex_has_its_own(tmp_path, monkeypatch):
    """Codex's fresh /tmp would hide a run directory under this machine's /tmp (SC: TMPDIR=/tmp/<user>)."""
    from types import SimpleNamespace
    from dycraft import environment_test as E
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    ctx = SimpleNamespace(tmp=Path("/tmp") / "dycraft_envtest_x", cleanup=[], args=SimpleNamespace(model="", effort=""),
                          facts={"codex_launch": {"wrapper": ["/w"], "tmpdir": ".tmp"}})
    probe = E.HarnessProbe(ctx, E.builtin_spec("codex"))
    assert probe.root.is_relative_to(tmp_path / "xdg" / "dycraft") and ctx.cleanup == [probe.root]
    ctx.facts["codex_launch"] = {"wrapper": [], "tmpdir": None}
    ctx.tmp = tmp_path / "plain"
    assert E.HarnessProbe(ctx, E.builtin_spec("codex")).root == tmp_path / "plain" / "harness_codex"


def test_a_write_the_reviewer_let_out_is_named_as_the_organizations_policy(tmp_path, monkeypatch):
    """SC, an account whose organization allows only on-request: the sandbox refused the outside write, the agent
    asked to run it outside, and Codex's reviewer said yes. The check names that, not a sandbox that failed."""
    import re
    from types import SimpleNamespace
    from dycraft import environment_test as E
    home = tmp_path / "codex2"
    _bundle(home, 'allowed_approval_policies = [ "on-request", "untrusted" ]\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = E.builtin_spec("codex")
    spec = E.HarnessSpec(**{**spec.__dict__, "config_dir": str(home), "binary": "/bin/true"})

    def launch(self, template, *, prompt, cwd, **kw):
        cwd.mkdir(parents=True, exist_ok=True)
        a, b = map(int, re.search(r"print\((\d+)\*(\d+)\)", prompt).groups())
        (cwd / "product.txt").write_text(f"{a * b}\n")
        Path(re.search(r"echo leaked > (\S+)", prompt).group(1)).write_text("leaked\n")
        return E.Ran(0, "DONE", "")
    monkeypatch.setattr(E.HarnessProbe, "launch", launch)
    ctx = SimpleNamespace(tmp=tmp_path / "envtest", cleanup=[], facts={},
                          args=E.parse_args(["--non-interactive", "--capabilities"]))
    checks = {c.id: c for c in E.register_backend_checks(ctx, spec)}
    assert checks["backend.codex.cap.shell"].fn(ctx).status == E.PASS
    result = checks["backend.codex.cap.confinement"].fn(ctx)
    assert result.status == E.WARN
    assert "through Codex's automatic reviewer: this account's organization allows only on-request, untrusted " \
           "approvals, so the reviewer can approve a command outside the sandbox" in result.summary
    assert result.fix == ["an account whose organization allows `never`, or `--backend claude`"]


def test_where_the_package_adds_no_profile_the_refusal_says_what_does(apparmor):
    """A stock AWS Ubuntu 24.04.5 image: /usr/bin/bwrap installed, and no AppArmor profile grants it user namespaces,
    so installing it again changes nothing."""
    apparmor(restricted=True, bwrap=True, profiled=False)
    text = B.codex_sandbox_refusal("the sandbox failed to start: bwrap: loopback: Failed RTM_NEWADDR", "/bin/codex",
                                   "/home/u/.codex", "codex-cli 0.167.0")
    assert "sudo apt-get install bubblewrap" not in text
    assert "  - With root: Give /usr/bin/bwrap an AppArmor profile that allows user namespaces (this system has " \
           "none, and the package alone does not add one): `printf 'abi <abi/4.0>," in text
    assert "  - With root: Or lift the restriction for every program: `sudo sysctl -w " \
           "kernel.apparmor_restrict_unprivileged_userns=0`" in text
    assert B.codex.codex_sandbox_fixes()[:2] == B.codex._userns_steps()
