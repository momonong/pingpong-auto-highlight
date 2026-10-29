"""Linux process accounting regressions; only bounded synthetic Python children."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pingpong_highlight import research_process_guard as media

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and wait4 required")


def waiting_child(root):
    """Hold the bounded fixture alive until the test injects its sampled exit."""
    release = root / "release-exit"
    code = (
        "import pathlib,sys,time; deadline=time.monotonic()+2; "
        "p=pathlib.Path(sys.argv[1]); "
        'exec("while not p.exists() and time.monotonic()<deadline: time.sleep(.001)")'
    )
    return [sys.executable, "-B", "-c", code, str(release)], release


class AccountingRegressions(unittest.TestCase):
    def budget(self, directory):
        return media.Budget(Path(directory), sorted(os.sched_getaffinity(0))[:2], total_seconds=3)

    def terminal(self, root, name):
        return json.loads((root / f"{name}.command.json").read_text())

    def test_natural_success_accounts_final_reads_without_zombie_proc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "payload.bin"
            payload.write_bytes(b"x" * 65536)
            code = (
                "import json,pathlib,sys; pathlib.Path(sys.argv[1]).read_byte"
                's(); value=int(dict(x.split(":",1) for x in pathlib.Path("/p'
                'roc/self/io").read_text().splitlines())["rchar"]); print(jso'
                'n.dumps({"observed_self_rchar":value}))'
            )
            budget = self.budget(root)
            budget.run(
                [sys.executable, "-B", "-c", code, str(payload)], root / "success", budget.now() + 2
            )
            child = json.loads((root / "success.stdout").read_text())
            entry = self.terminal(root, "success")
            self.assertEqual(entry["status"], "succeeded")
            self.assertTrue(entry["reaped"])
            self.assertTrue(entry["accounting_complete"])
            self.assertGreaterEqual(entry["logical_read_rchar_bytes"], child["observed_self_rchar"])
            self.assertIsNone(entry["logical_read_rchar_exact_child_bytes"])
            self.assertEqual(entry["rchar_method"], "conservative_parent_self_io_delta_after_wait4")
            self.assertGreater(entry["wait4_rusage"]["maxrss_bytes"], 0)

    def test_timeout_preserves_primary_reason_and_unrelated_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            survivor = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time;time.sleep(2)"], start_new_session=True
            )
            try:
                budget = self.budget(root)
                with self.assertRaises(media.GuardError):
                    budget.run(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            'import time;print("ready",flush=True);time.sleep(2)',
                        ],
                        root / "timeout",
                        budget.now() + 0.15,
                    )
                entry = self.terminal(root, "timeout")
                self.assertEqual(entry["stop_reason"], "clip_wall_limit")
                self.assertEqual(entry["returncode"], -signal.SIGKILL)
                self.assertTrue(entry["reaped"])
                self.assertTrue(entry["owned_group_confirmed"])
                self.assertIsNone(survivor.poll())
            finally:
                survivor.terminate()
                survivor.wait(timeout=1)

    def run_monitor_fault(self, fault, label):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = media.proc_stats

            def fail_child(pid):
                if pid != os.getpid():
                    raise fault
                return real(pid)

            budget = self.budget(root)
            with (
                patch.object(media, "proc_stats", side_effect=fail_child),
                self.assertRaises(media.GuardError),
            ):
                budget.run(
                    [sys.executable, "-B", "-c", "import time;time.sleep(2)"],
                    root / label,
                    budget.now() + 1,
                )
            entry = self.terminal(root, label)
            self.assertEqual(entry["stop_reason"], "live_monitor_error")
            self.assertEqual(entry["status"], "rejected")
            self.assertTrue(entry["reaped"])
            self.assertIn(type(fault).__name__, entry["errors"][0]["type"])
            with self.assertRaises(media.GuardError):
                budget.run(
                    [sys.executable, "-B", "-c", "pass"], root / "forbidden", budget.now() + 1
                )
            self.assertFalse((root / "forbidden.stdout").exists())

    def test_live_eacces_is_rejected_and_durably_accounted(self):
        self.run_monitor_fault(PermissionError("synthetic live EACCES"), "denied")

    def test_missing_live_process_is_rejected(self):
        self.run_monitor_fault(FileNotFoundError("synthetic missing live proc"), "missing")

    def test_malformed_or_missing_proc_io_is_rejected(self):
        for raw in ("rchar: abc\nread_bytes: 0\n", "rchar: -1\nread_bytes: 0\n", "rchar: 12\n"):
            with self.assertRaises(media.GuardError):
                media.parse_io(raw)

    def test_final_accounting_fault_never_becomes_zero_or_success(self):
        self.final_fault(False)

    def test_final_accounting_fault_keeps_timeout_as_primary(self):
        self.final_fault(True)

    def final_fault(self, timeout):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real_wait4 = os.wait4
            real_read = media.read_io
            reaped = False

            def wait4(pid, flags):
                nonlocal reaped
                result = real_wait4(pid, flags)
                if result[0] == pid:
                    reaped = True
                return result

            def read_io(pid):
                if pid == os.getpid() and reaped:
                    raise PermissionError("synthetic final accounting EACCES")
                return real_read(pid)

            budget = self.budget(root)
            delay = 2 if timeout else 0.05
            with (
                patch.object(media.os, "wait4", side_effect=wait4),
                patch.object(media, "read_io", side_effect=read_io),
                self.assertRaises(media.GuardError),
            ):
                budget.run(
                    [sys.executable, "-B", "-c", f"import time;time.sleep({delay})"],
                    root / "final-fault",
                    budget.now() + (0.15 if timeout else 1),
                )
            entry = self.terminal(root, "final-fault")
            self.assertEqual(
                entry["stop_reason"], "clip_wall_limit" if timeout else "final_accounting_error"
            )
            self.assertTrue(entry["reaped"])
            self.assertFalse(entry["accounting_complete"])
            self.assertIsNone(entry["logical_read_rchar_bytes"])
            self.assertIsNone(budget.snapshot()["logical_read_rchar_bytes"])
            self.assertTrue(any(error["phase"] == "final_accounting" for error in entry["errors"]))

    def test_cleanup_fault_does_not_erase_timeout_or_final_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real_killpg = os.killpg

            def kill_then_report_error(pgid, sig):
                real_killpg(pgid, sig)
                raise OSError("synthetic cleanup reporting failure after SIGKILL")

            budget = self.budget(root)
            try:
                with (
                    patch.object(media.os, "killpg", side_effect=kill_then_report_error),
                    self.assertRaises(media.GuardError),
                ):
                    budget.run(
                        [sys.executable, "-B", "-c", "import time;time.sleep(2)"],
                        root / "cleanup-fault",
                        budget.now() + 0.15,
                    )
            finally:
                if budget.commands and "pid" in budget.commands[-1]:
                    with suppress(ChildProcessError):
                        os.waitpid(budget.commands[-1]["pid"], os.WNOHANG)
            entry = self.terminal(root, "cleanup-fault")
            self.assertEqual(entry["stop_reason"], "clip_wall_limit")
            self.assertTrue(entry["reaped"])
            self.assertTrue(any(error["phase"] == "cleanup" for error in entry["errors"]))

    def test_late_cleanup_census_failure_still_signals_pinned_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = media.owned_group_members
            post_exit_calls = 0

            def fail_late(token):
                nonlocal post_exit_calls
                terminal = os.waitid(os.P_PID, token["pid"], os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if terminal is not None:
                    post_exit_calls += 1
                    if post_exit_calls >= 2:
                        raise PermissionError("synthetic late persistent cleanup census EACCES")
                return real(token)

            budget = self.budget(root)
            with (
                patch.object(media, "owned_group_members", side_effect=fail_late),
                self.assertRaises(media.GuardError),
            ):
                budget.run(
                    [sys.executable, "-B", "-c", "pass"], root / "late-census", budget.now() + 1
                )
            entry = self.terminal(root, "late-census")
            self.assertEqual(entry["status"], "rejected")
            self.assertEqual(entry["stop_reason"], "cleanup_census_error")
            self.assertEqual(entry["termination_signal_sent"], "SIGKILL")
            self.assertEqual(entry["returncode"], 0)
            self.assertTrue(entry["reaped"])
            self.assertFalse(entry["descendant_scope_known"])
            self.assertFalse(entry["accounting_complete"])
            self.assertIsNone(entry["logical_read_rchar_bytes"])

    def persistent_census_failure(self, fault):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            survivor = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time;time.sleep(2)"], start_new_session=True
            )
            budget = self.budget(root)
            cleanup = {"manual_cleanup_needed": False}
            try:
                with (
                    patch.object(media, "owned_group_members", side_effect=fault),
                    self.assertRaises(media.GuardError),
                ):
                    budget.run(
                        [sys.executable, "-B", "-c", "import time;time.sleep(2)"],
                        root / "census-fault",
                        budget.now() + 1,
                    )
                entry = self.terminal(root, "census-fault")
                self.assertIsNone(survivor.poll())
                self.assertEqual(entry["status"], "rejected")
                self.assertEqual(entry["stop_reason"], "live_monitor_error")
                self.assertTrue(
                    entry["reaped"],
                    "persistent unrelated census failure prevented owned leader reap",
                )
                self.assertEqual(entry["returncode"], -signal.SIGKILL)
                self.assertTrue(entry["owned_group_confirmed"])
                self.assertFalse(entry["descendant_scope_known"])
                self.assertFalse(entry["accounting_complete"])
                self.assertIsNone(entry["logical_read_rchar_bytes"])
                self.assertIsNone(entry["live_owned_group_after_cleanup"])
            finally:
                if budget.commands and "pid" in budget.commands[-1]:
                    pid = budget.commands[-1]["pid"]
                    try:
                        state = media.identity(pid)
                        self.assertEqual(state["ppid"], os.getpid())
                        self.assertEqual(state["pgrp"], pid)
                        self.assertEqual(state["session"], pid)
                        cleanup["manual_cleanup_needed"] = True
                        if state["state"] != "Z":
                            os.killpg(pid, signal.SIGKILL)
                        waited, status = os.waitpid(pid, 0)
                        cleanup.update(
                            {
                                "pid": waited,
                                "returncode": os.waitstatus_to_exitcode(status),
                                "fixture_reaped": True,
                            }
                        )
                    except FileNotFoundError:
                        cleanup["fixture_reaped"] = True
                survivor.terminate()
                survivor.wait(timeout=1)

    def test_persistent_census_permission_failure_still_reaps_owned_child(self):
        self.persistent_census_failure(
            PermissionError("synthetic persistent unrelated proc census EACCES")
        )

    def test_persistent_census_parse_failure_still_reaps_owned_child(self):
        self.persistent_census_failure(
            media.GuardError("synthetic persistent unrelated proc stat malformed")
        )

    def parent_sample_failure_after_child_exit(self, fault):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            release = root / "release-child"
            code = (
                'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); exec("'
                'while not p.exists(): time.sleep(.001)")'
            )
            real = media.proc_stats

            def sample(pid):
                if pid == os.getpid():
                    raise fault
                result = real(pid)
                release.write_text("exit")
                os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
                return result

            budget = self.budget(root)
            caught = False
            with patch.object(media, "proc_stats", side_effect=sample):
                try:
                    budget.run(
                        [sys.executable, "-B", "-c", code, str(release)],
                        root / "parent-fault",
                        budget.now() + 1,
                    )
                except media.GuardError:
                    caught = True
            entry = self.terminal(root, "parent-fault")
            self.assertTrue(
                caught, "still-live parent monitoring fault was swallowed after child exit"
            )
            self.assertEqual(entry["status"], "rejected")
            self.assertEqual(entry["stop_reason"], "live_monitor_error")
            self.assertEqual(entry["returncode"], 0)
            self.assertTrue(entry["reaped"])
            self.assertTrue(entry["accounting_complete"])
            self.assertTrue(any(error["type"] == type(fault).__name__ for error in entry["errors"]))

    def test_parent_permission_error_is_not_child_exit_race(self):
        self.parent_sample_failure_after_child_exit(
            PermissionError("synthetic still-live parent EACCES")
        )

    def test_parent_missing_proc_is_not_child_exit_race(self):
        self.parent_sample_failure_after_child_exit(
            FileNotFoundError("synthetic still-live parent missing proc")
        )

    def test_typed_exit_race_requires_real_wait_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = media.proc_stats
            command, release = waiting_child(root)

            def exit_race(pid):
                if pid != os.getpid():
                    release.write_text("exit")
                    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
                    raise media.ProcessExited("synthetic VmRSS disappeared after confirmed exit")
                return real(pid)

            budget = self.budget(root)
            with patch.object(media, "proc_stats", side_effect=exit_race):
                budget.run(command, root / "typed-exit", budget.now() + 1)
            entry = self.terminal(root, "typed-exit")
            self.assertEqual(entry["status"], "succeeded")
            self.assertEqual(entry["exit_race_reconciled"], "ProcessExited")
            self.assertTrue(entry["accounting_complete"])

    def test_malformed_sample_is_rejected_even_when_child_has_exited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = media.proc_stats
            command, release = waiting_child(root)

            def malformed(pid):
                if pid != os.getpid():
                    release.write_text("exit")
                    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
                    raise media.GuardError("synthetic malformed counter at exit")
                return real(pid)

            budget = self.budget(root)
            with (
                patch.object(media, "proc_stats", side_effect=malformed),
                self.assertRaises(media.GuardError),
            ):
                budget.run(command, root / "malformed-exit", budget.now() + 1)
            entry = self.terminal(root, "malformed-exit")
            self.assertEqual(entry["status"], "rejected")
            self.assertEqual(entry["stop_reason"], "live_monitor_error")
            self.assertTrue(entry["reaped"])

    def test_exited_leader_descendant_is_stopped_and_accounting_incomplete(self):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            survivor = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time;time.sleep(2)"], start_new_session=True
            )
            grandchild = None
            try:
                code = (
                    "import os,time; pid=os.fork(); print(pid,flush=True) if pid "
                    "else None; os._exit(0) if pid else time.sleep(2)"
                )
                budget = self.budget(root)
                real_census = media.owned_group_members
                synchronized = False

                def census_after_leader_exit(token):
                    nonlocal synchronized
                    if not synchronized:
                        deadline = budget.now() + 0.5
                        while True:
                            event = os.waitid(
                                os.P_PID, token["pid"], os.WEXITED | os.WNOHANG | os.WNOWAIT
                            )
                            if event is not None:
                                break
                            if budget.now() >= deadline:
                                raise AssertionError(
                                    "fixture leader did not exit before first census"
                                )
                            time.sleep(0.001)
                        self.assertEqual(event.si_pid, token["pid"])
                        self.assertEqual(event.si_status, 0)
                        leader = media.identity(token["pid"])
                        self.assertEqual(leader["state"], "Z")
                        members = real_census(token)
                        live_descendants = [
                            item
                            for item in members
                            if item["pid"] != token["pid"] and item["state"] != "Z"
                        ]
                        self.assertEqual(len(live_descendants), 1)
                        precondition = {
                            "method": (
                                "bounded waitid WEXITED|WNOHANG|WNOWAIT before first real census"
                            ),
                            "leader_pid": event.si_pid,
                            "leader_status": event.si_status,
                            "leader_kernel_state": leader["state"],
                            "live_descendants_before_monitor_cleanup": live_descendants,
                        }
                        (root / "fixture-precondition.json").write_text(
                            json.dumps(precondition, indent=2) + "\n"
                        )
                        synchronized = True
                        return members
                    return real_census(token)

                with (
                    patch.object(
                        media, "owned_group_members", side_effect=census_after_leader_exit
                    ),
                    self.assertRaises(media.GuardError),
                ):
                    budget.run(
                        [sys.executable, "-B", "-c", code], root / "orphan", budget.now() + 1
                    )
                grandchild = int((root / "orphan.stdout").read_text().splitlines()[0])
                self.assertTrue(synchronized)
                (root / "fixture-observation.json").write_text(
                    json.dumps(
                        {
                            "synchronization_completed": synchronized,
                            "sentinel_returncode_after_cleanup": survivor.poll(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                entry = self.terminal(root, "orphan")
                self.assertEqual(entry["returncode"], 0)
                self.assertTrue(entry["leader_exited_before_descendant_cleanup"])
                self.assertEqual(entry["stop_reason"], "unexpected_descendant_process")
                self.assertFalse(entry["accounting_complete"])
                self.assertIsNone(entry["logical_read_rchar_bytes"])
                self.assertTrue(entry["reaped"])
                self.assertTrue(
                    any(member["pid"] == grandchild for member in entry["unexpected_descendants"])
                )
                self.assertTrue(
                    all(member["state"] == "Z" for member in entry["owned_group_after_cleanup"])
                )
                self.assertIsNone(survivor.poll())
                pid, status = os.waitpid(grandchild, os.WNOHANG)
                self.assertEqual(pid, grandchild)
                self.assertTrue(os.WIFSIGNALED(status))
                self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
                grandchild = None
            finally:
                if grandchild:
                    try:
                        os.kill(grandchild, signal.SIGKILL)
                        os.waitpid(grandchild, 0)
                    except ProcessLookupError:
                        pass
                survivor.terminate()
                survivor.wait(timeout=1)
                self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)


def test_child_inherits_callers_directory_instead_of_private_checkout(tmp_path, monkeypatch):
    """Public extraction must never force a private repository working directory."""
    monkeypatch.chdir(tmp_path)
    budget = media.Budget(tmp_path, sorted(os.sched_getaffinity(0))[:1], total_seconds=3)
    output = budget.run(
        [sys.executable, "-B", "-c", "import os;print(os.getcwd())"],
        tmp_path / "cwd",
        budget.now() + 1,
    )
    assert output.read_text().strip() == str(tmp_path)


def test_existing_or_symlink_command_receipt_prevents_dispatch(tmp_path):
    journal = tmp_path / "duplicate.command.json"
    journal.symlink_to(tmp_path / "absent-target")
    budget = media.Budget(tmp_path, sorted(os.sched_getaffinity(0))[:1])
    with pytest.raises(media.GuardError):
        budget.run([sys.executable, "-B", "-c", "pass"], tmp_path / "duplicate", budget.now() + 1)
    assert journal.is_symlink()
    assert not (tmp_path / "duplicate.stdout").exists()


@pytest.mark.parametrize("exception", [PermissionError, FileNotFoundError])
def test_child_permission_and_missing_exit_races_require_kernel_confirmation(tmp_path, exception):
    real = media.proc_stats
    command, release = waiting_child(tmp_path)

    def confirmed_exit(pid):
        if pid != os.getpid():
            release.write_text("exit")
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
            raise exception("synthetic confirmed exit race")
        return real(pid)

    budget = media.Budget(tmp_path, sorted(os.sched_getaffinity(0))[:1])
    with patch.object(media, "proc_stats", side_effect=confirmed_exit):
        budget.run(
            command,
            tmp_path / "exit-race",
            budget.now() + 1,
        )
    assert budget.commands[-1]["status"] == "succeeded"
    assert budget.commands[-1]["exit_race_reconciled"] == exception.__name__


def test_typed_exit_signal_for_live_child_is_rejected(tmp_path):
    test = AccountingRegressions()
    test.run_monitor_fault(media.ProcessExited("synthetic unconfirmed exit"), "live-exit")


@pytest.mark.parametrize("phase", ["initial_census", "cleanup_poll_census"])
def test_one_time_census_descendant_is_retained_and_rejects_further_dispatch(tmp_path, phase):
    """An observed descendant remains accounting evidence after later censuses omit it."""
    leader = {
        "pid": 987654321,
        "state": "Z",
        "ppid": os.getpid(),
        "pgrp": 987654321,
        "session": 987654321,
        "start_ticks": 1,
        "cpu_ticks": 0,
    }
    descendant = {**leader, "pid": 987654322, "ppid": leader["pid"], "state": "S"}
    samples = (
        [[leader, descendant], [leader], [leader]]
        if phase == "initial_census"
        else [[leader], [leader], [leader, descendant], [leader]]
    )
    usage = SimpleNamespace(ru_utime=0, ru_stime=0, ru_maxrss=1, ru_inblock=0, ru_oublock=0)
    process = SimpleNamespace(pid=leader["pid"], returncode=None)
    journal = []
    with (
        patch.object(
            media, "save", side_effect=lambda path, entry: journal.append(copy.deepcopy(entry))
        ),
        patch.object(media, "identity", return_value=leader),
        patch.object(media, "owned_group_members", side_effect=samples),
        patch.object(
            media,
            "read_io",
            side_effect=[{"rchar": 0, "read_bytes": 0}, {"rchar": 10, "read_bytes": 0}],
        ),
        patch.object(media.subprocess, "Popen", return_value=process) as dispatch,
        patch.object(media.os, "waitid", return_value=object()),
        patch.object(media.os, "wait4", return_value=(leader["pid"], 0, usage)),
        patch.object(media.os, "killpg") as signals,
        patch.object(media.os.path, "lexists", return_value=False),
        patch.object(Path, "open", return_value=MagicMock()),
        patch.object(media.time, "sleep"),
    ):
        budget = media.Budget(tmp_path, [0], total_seconds=3)
        with suppress(media.GuardError):
            budget.run(["synthetic-command"], tmp_path / phase, budget.now() + 1)
        entry = journal[-1]
        assert entry["status"] == "rejected"
        assert entry["stop_reason"] == "unexpected_descendant_process"
        assert entry["accounting_complete"] is False
        assert entry["logical_read_rchar_bytes"] is None
        assert entry["unexpected_descendants"] == [descendant]
        assert entry["owned_identity"] == leader
        assert entry["reaped"] is True
        assert entry["returncode"] == 0
        assert budget.failed is True
        assert budget.accounting_complete is False
        signals.assert_called_with(leader["pid"], signal.SIGKILL)
        with pytest.raises(media.GuardError, match="no further child dispatch"):
            budget.run(["synthetic-forbidden"], tmp_path / "forbidden", budget.now() + 1)
        assert len(budget.commands) == 1
        assert dispatch.call_count == 1
