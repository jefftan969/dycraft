"""The sandbox binds only the run's GPU nodes, and `CUDA_VISIBLE_DEVICES` inside
it matches them (`runtime._nvidia_selection`). EGL ignores the variable, so on a
multi-GPU node without per-job isolation, binding every node would let EEVEE
render on GPU 0 under `CUDA_VISIBLE_DEVICES=5`."""
from pathlib import Path

import pytest

from dycraft.animation.blender import runtime

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")


@pytest.fixture
def gpus(monkeypatch):
    """Pretend the host has eight GPU nodes, of which `open` is the set this
    process may open (a slurm device cgroup refuses the others with EPERM)."""
    def setup(opened, visible=None):
        nodes = [Path(f"/dev/nvidia{i}") for i in range(8)]
        monkeypatch.setattr(runtime, "_dev_nodes", lambda: nodes)
        monkeypatch.setattr(runtime, "_openable", lambda dev: int(dev.name[len("nvidia"):]) in opened)
        monkeypatch.setattr(runtime, "_NVIDIA_USABLE", None)
        if visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    return setup


def names(nodes):
    return [n.name for n in nodes]


def test_without_isolation_an_index_list_binds_those_nodes_and_is_renumbered(gpus):
    gpus(opened=set(range(8)), visible="5")
    assert names(runtime._nvidia_devices()) == ["nvidia5"]
    assert runtime.worker_env()["CUDA_VISIBLE_DEVICES"] == "0"
    gpus(opened=set(range(8)), visible="3,1")
    assert names(runtime._nvidia_devices()) == ["nvidia1", "nvidia3"]
    assert runtime.worker_env()["CUDA_VISIBLE_DEVICES"] == "1,0"   # the operator's order, in sandbox numbers


def test_under_a_device_cgroup_the_granted_nodes_are_bound_and_the_variable_passes_verbatim(gpus):
    # slurm numbers the variable within the grant: "0" is the one GPU granted,
    # whichever node that is; picking /dev/nvidia0 by index would bind a node
    # the job may not open
    gpus(opened={5}, visible="0")
    assert names(runtime._nvidia_devices()) == ["nvidia5"]
    env = runtime.worker_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert runtime.nvidia_usable()


def test_no_gpu_granted_binds_none_and_is_not_usable(gpus):
    gpus(opened=set())
    assert runtime._nvidia_devices() == []
    assert not runtime.nvidia_usable()
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in runtime.worker_env()


@pytest.mark.parametrize("visible", [None, "", "GPU-1a2b,2", "-1"])
def test_other_values_keep_every_openable_node_and_pass_verbatim(gpus, visible):
    gpus(opened=set(range(8)), visible=visible)
    assert len(runtime._nvidia_devices()) == 8
    env = runtime.worker_env()
    if visible is None:
        assert "CUDA_VISIBLE_DEVICES" not in env
    else:
        assert env["CUDA_VISIBLE_DEVICES"] == visible


def test_the_sandbox_binds_the_arms_node_and_the_control_nodes_only(gpus, monkeypatch, tmp_path):
    gpus(opened=set(range(8)), visible="5")
    dev = [Path("/dev/dri"), Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm")] + [Path(f"/dev/nvidia{i}") for i in range(8)]
    real_glob, real_exists = Path.glob, Path.exists
    monkeypatch.setattr(Path, "glob", lambda self, pat: (iter([p for p in dev if p.name.startswith("nvidia")])
                                                         if str(self) == "/dev" and pat == "nvidia*" else real_glob(self, pat)))
    monkeypatch.setattr(Path, "exists", lambda self: True if self in dev else real_exists(self))
    try:
        cmd = runtime.command(tmp_path / "ws", tmp_path / "out", tmp_path / "job.json")
    except RuntimeError as e:      # no bwrap or coal interpreter on this host
        pytest.skip(str(e))
    bound = {cmd[i + 1] for i, a in enumerate(cmd) if a == "--dev-bind"}
    assert {"/dev/nvidia5", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/dri"} <= bound
    assert not any(b.startswith("/dev/nvidia") and b[len("/dev/nvidia"):].isdigit() and b != "/dev/nvidia5" for b in bound)


def _ik_child_gpu(monkeypatch, tmp_path):
    """The GPU variables the real `h.ik` child sees, with this interpreter
    standing in for the cuRobo one."""
    import importlib
    import sys
    ik_mod = importlib.import_module("dycraft.animation.ik.ik")
    monkeypatch.setattr(ik_mod, "_resolve_ik_python", lambda: sys.executable)
    monkeypatch.setattr(ik_mod, "gpu_visible", lambda: (True, ""))
    # The setup's no-GPU backstop would refuse this stand-in; it has its own test.
    no_setup = tmp_path / "no_setup.py"
    no_setup.parent.mkdir(parents=True, exist_ok=True)
    no_setup.write_text("")
    monkeypatch.setattr(ik_mod, "_SETUP_PATH", no_setup)
    out = ik_mod.ik("import os\nresult = {k: os.environ.get(k) for k in "
                    "('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}",
                    ik_dir=tmp_path, require_scene=False)
    assert out["ok"], out
    return out["result"]


def test_h_ik_and_the_sandbox_name_the_same_card(gpus, monkeypatch, tmp_path):
    """The IK child runs outside the sandbox and sees every node, so it keeps the
    absolute index; both number in PCI order, the order of the nodes. CUDA's
    default order (fastest first) can put another card at 5 on a node of mixed
    GPUs while Blender renders on /dev/nvidia5."""
    gpus(opened=set(range(8)), visible="5")
    assert names(runtime._nvidia_devices()) == ["nvidia5"]
    env = runtime.worker_env()
    assert (env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"]) == ("0", "PCI_BUS_ID")
    assert _ik_child_gpu(monkeypatch, tmp_path) == {"CUDA_VISIBLE_DEVICES": "5",
                                                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_h_ik_keeps_an_operator_s_order_and_an_unset_gpu(gpus, monkeypatch, tmp_path):
    gpus(opened=set(range(8)), visible="GPU-1a2b")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert _ik_child_gpu(monkeypatch, tmp_path) == {"CUDA_VISIBLE_DEVICES": "GPU-1a2b",
                                                    "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    gpus(opened=set(range(8)))
    assert _ik_child_gpu(monkeypatch, tmp_path / "unset") == {"CUDA_VISIBLE_DEVICES": None,
                                                              "CUDA_DEVICE_ORDER": None}


def test_worker_caches_optix_kernels_and_isolates_blender_user_files(monkeypatch):
    """The OptiX and CUDA kernel caches default to places the sandbox does not
    keep (`/var/tmp/OptixCache_<user>`, `~/.nv` under its `/tmp`), so they go under
    the persistent cache mount; Blender's user resources are named and fresh in
    every process (the sandbox's `/tmp`)."""
    monkeypatch.setattr(runtime, "cache_dir", lambda: "/host/cache")
    env = runtime.worker_env()
    assert env["OPTIX_CACHE_PATH"] == runtime.CACHE_MOUNT + "/optix"
    assert env["CUDA_CACHE_PATH"] == runtime.CACHE_MOUNT + "/nvidia/ComputeCache"
    assert env["BLENDER_USER_RESOURCES"] == "/tmp/blender_user"
    monkeypatch.setattr(runtime, "cache_dir", lambda: None)
    env = runtime.worker_env()
    assert "OPTIX_CACHE_PATH" not in env       # nowhere persistent to put it
    assert "CUDA_CACHE_PATH" not in env
    assert env["BLENDER_USER_RESOURCES"] == "/tmp/blender_user"


def test_the_cache_dir_makes_the_kernel_caches_and_survives_one_it_cannot(monkeypatch, tmp_path):
    """`cache_dir` makes the OptiX and CUDA kernel-cache directories itself; a
    stray file where one should go loses that cache only, not the shader cache."""
    root = tmp_path / "cache"
    monkeypatch.setenv("DYCRAFT_BLENDER_CACHE", str(root))
    assert runtime.cache_dir() == root
    assert (root / "optix").is_dir() and (root / "nvidia" / "ComputeCache").is_dir()
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "optix").write_text("not a directory")
    monkeypatch.setenv("DYCRAFT_BLENDER_CACHE", str(blocked))
    assert runtime.cache_dir() == blocked
    assert (blocked / "nvidia" / "ComputeCache").is_dir()
