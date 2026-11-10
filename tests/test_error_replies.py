"""A failed reply names one of four classes, and only a harness fault points at a log.

`rejection` and `check_finding` are set where the harness refuses or reports;
these tests cover the two an exception decides: `author_code`, raised inside
the agent's own code (`worker.authored`, exercised in `test_resident_blender`),
and `harness`, anything else, which is one line and a log path rather than a
traceback the agent reads as something to repair in the scene.
"""
from __future__ import annotations

from dycraft.agent.server import _handler_failure
from dycraft.animation.blender.runtime import WorkerError, failed_result, worker_failure


def test_a_worker_s_class_is_carried_and_only_a_harness_fault_is_framed(tmp_path):
    own = worker_failure({"ok": False, "failure_class": "author_code",
                          "error": "Traceback ...\nKeyError: 'Cube'"}, tmp_path / "result.json")
    assert own.failure_class == "author_code" and str(own).endswith("KeyError: 'Cube'")
    fault = worker_failure({"ok": False, "failure_class": "harness",
                            "error": "AttributeError: 'NoneType' object has no attribute 'x'"},
                           tmp_path / "result.json")
    assert fault.failure_class == "harness"
    assert str(fault).startswith("harness fault: AttributeError")
    assert str(fault).endswith(f"Log: {tmp_path / 'result.json'}")
    assert fault.reason.startswith("AttributeError")
    assert failed_result(own, tmp_path)["failure_class"] == "author_code"


def test_an_exception_raised_in_the_harness_is_a_refusal_or_a_fault(tmp_path):
    refused = failed_result(ValueError("unknown shot 'x'"), tmp_path)
    assert refused == {"ok": False, "error": "unknown shot 'x'", "failure_class": "rejection"}
    try:
        {}["missing"]
    except KeyError as error:
        fault = failed_result(error, tmp_path)
    assert fault["failure_class"] == "harness"
    assert fault["error"].startswith("harness fault: KeyError: 'missing'")
    assert "Traceback" in (tmp_path / "error.txt").read_text()
    # A canceled job is the agent's own request: its status says so.
    assert "failure_class" not in failed_result(RuntimeError("job cancelled"), tmp_path, True)
    worker = WorkerError("Blender worker exited -9 during render; ...", "harness")
    assert failed_result(worker, tmp_path)["error"] == str(worker)


def test_an_sdk_error_is_the_author_s_code_not_a_harness_fault(tmp_path):
    """An SDK error raised by the author's own code (here a reference to a part
    it named) is the author's to repair: as a harness fault it would tell the
    agent not to change the scene for it."""
    from dycraft.sdk._errors import DanglingReference
    error = DanglingReference("support_connected names 'arm/rest', which is not a part",
                              suggested_repairs=["load the object through load_object"])
    for reply in (failed_result(error, tmp_path), _handler_failure(error, tmp_path, "build")):
        assert reply["failure_class"] == "author_code"
        assert reply["error"].startswith("support_connected names")
        assert reply["error_detail"]["code"] == "DanglingReference"
        assert reply["error_detail"]["suggested_repairs"] == ["load the object through load_object"]
    assert not (tmp_path / "error.txt").exists()


def test_a_raising_handler_replies_in_one_line(tmp_path):
    assert _handler_failure(ValueError("wait for pending edits"), tmp_path, "blender") == {
        "error": "wait for pending edits", "failure_class": "rejection"}
    try:
        None.x
    except AttributeError as error:
        reply = _handler_failure(error, tmp_path, "compile")
    assert reply["failure_class"] == "harness" and "\n" not in reply["error"]
    assert "compile" in (tmp_path / "harness_errors.log").read_text()
