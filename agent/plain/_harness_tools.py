# --- plain-layer tool stubs (appended to _harness_core.py at stage time) ---
# The plain layer exposes exactly one harness op: submit_result. Everything
# else the agent needs (reading files, running code) it does through its own
# native CLI tools (bash/edit), sandboxed to the run dir. `_call` is defined
# in the concatenated transport core above.


def submit_result(text: str):
    """Return your final answer for this one-off task as free text.

    This is the only way to deliver a result. Calling it
    records `<run_dir>/result.json`, marks the run complete, and terminates
    the session — so call it exactly once, last, with your complete answer."""
    return _call("submit_result", {"text": text})
