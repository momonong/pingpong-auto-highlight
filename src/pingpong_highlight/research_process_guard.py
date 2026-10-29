"""Experimental Linux process guard with conservative observed accounting.

Not wired into the product. Uses /proc, isolated child sessions and wait4. A
complete receipt covers observed direct-child reads, not a hard subtree bound;
very short or escaped descendants can evade a sampled census. Call serially from
a single-threaded supervisor (Popen uses preexec_fn), with a dedicated output root.
The caller owns command authorization; this module does not grant media admission.
"""

import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

GiB = 1024**3
MiB = 1024**2


class GuardError(RuntimeError):
    pass


class ProcessExited(GuardError):
    """Typed lifecycle transition; accepted only after waitid confirms exit."""

    pass


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def utc():
    return datetime.now(UTC).isoformat()


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def parse_io(raw):
    """Reject missing, duplicate, negative or malformed kernel counters."""
    try:
        counters = {}
        for line in raw.splitlines():
            key, value = line.split(":", 1)
            if key in counters:
                raise ValueError("duplicate counter")
            counters[key] = int(value.strip())
            if counters[key] < 0:
                raise ValueError("negative counter")
        for key in ("rchar", "read_bytes"):
            if key not in counters:
                raise ValueError(f"missing {key}")
        return counters
    except (ValueError, TypeError) as error:
        raise GuardError(f"malformed /proc io: {error}") from error


def read_io(pid):
    name = "self" if pid == os.getpid() else str(pid)
    return parse_io(Path(f"/proc/{name}/io").read_text())


def identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "state": raw[0],
        "ppid": int(raw[1]),
        "pgrp": int(raw[2]),
        "session": int(raw[3]),
        "start_ticks": int(raw[19]),
        "cpu_ticks": int(raw[11]) + int(raw[12]),
    }


def proc_stats(pid):
    # No None/zero fallback: the caller must reconcile an exit race or reject.
    io = read_io(pid)
    state = identity(pid)
    if state["state"] == "Z":
        raise ProcessExited("process became zombie before RSS sampling")
    status = dict(
        line.split(":", 1)
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        if ":" in line
    )
    if "VmRSS" not in status:
        if identity(pid)["state"] == "Z":
            raise ProcessExited("VmRSS disappeared during process exit")
        raise GuardError("missing live VmRSS")
    rss = int(status["VmRSS"].split()[0]) * 1024
    if rss < 0:
        raise GuardError("negative live VmRSS")
    return {**io, **state, "rss": rss}


def confirm_owned_group(token):
    """Verify only the pinned owned leader; no unrelated /proc census dependency."""
    leader = identity(token["pid"])
    for key in ("pid", "ppid", "pgrp", "session", "start_ticks"):
        if leader[key] != token[key]:
            raise GuardError("owned leader identity changed; group signal forbidden")
    if (
        token["pgrp"] != token["pid"]
        or token["session"] != token["pid"]
        or token["ppid"] != os.getpid()
    ):
        raise GuardError("process is not this caller's isolated child group")
    return leader


def owned_group_members(token):
    """Census failure invalidates descendant knowledge, not pinned leader ownership."""
    confirm_owned_group(token)
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            item = identity(int(directory.name))
        except FileNotFoundError:
            continue  # A vanished unrelated process is no longer a live group member.
        if item["pgrp"] == token["pgrp"] and item["session"] == token["session"]:
            members.append(item)
    return members


class Budget:
    """Experimental serial direct-child accounting on Linux.

    Final /proc/self/io after wait4 includes waited children and monitor overhead.
    It is conservative observed direct-child accounting, not exact child/source
    bytes or a hard whole-subtree bound. Unwaited descendants invalidate success.
    """

    def __init__(self, root, cores, total_seconds=120):
        self.root, self.cores = Path(root), list(cores)
        self.started = time.monotonic()
        self.total_seconds = total_seconds
        self.rchar = self.read_bytes = self.cpu_seconds = 0
        self.peak_rss = self.child_runtime = 0
        self.commands = []
        self.poll_seconds = 0.02
        self.overshoots = []
        self.failed = False
        self.accounting_complete = True

    @staticmethod
    def now():
        return time.monotonic()

    def snapshot(self):
        return {
            "elapsed_wall_seconds": self.now() - self.started,
            "logical_read_rchar_bytes": self.rchar if self.accounting_complete else None,
            "known_conservative_rchar_charge_bytes": self.rchar,
            "accounting_complete": self.accounting_complete,
            "physical_read_bytes_observed": self.read_bytes if self.accounting_complete else None,
            "cpu_seconds_wait4": self.cpu_seconds,
            "cumulative_child_runtime_seconds": self.child_runtime,
            "peak_aggregate_rss_sample_bytes": self.peak_rss,
            "output_tree_bytes_at_snapshot": tree_bytes(self.root),
            "poll_interval_seconds": self.poll_seconds,
            "overshoots": self.overshoots,
        }

    def run(self, command, prefix, clip_deadline):
        if self.failed or not self.accounting_complete:
            raise GuardError("budget already rejected/incomplete; no further child dispatch")
        if self.now() >= clip_deadline or self.now() - self.started >= self.total_seconds:
            raise GuardError("wall budget exhausted before child start")
        prefix = Path(prefix)
        journal = prefix.with_suffix(".command.json")
        if any(
            os.path.lexists(prefix.with_suffix(suffix))
            for suffix in (".command.json", ".stdout", ".stderr")
        ):
            raise GuardError("existing command output; no overwrite or retry")
        entry = {
            "operation_id": str(uuid.uuid4()),
            "command": list(map(str, command)),
            "started_utc": utc(),
            "status": "started",
            "stop_reason": None,
            "stdout": str(prefix.with_suffix(".stdout").relative_to(self.root)),
            "stderr": str(prefix.with_suffix(".stderr").relative_to(self.root)),
            "errors": [],
            "returncode": None,
            "reaped": False,
            "owned_group_confirmed": False,
            "descendant_scope_known": False,
            "accounting_complete": False,
            "logical_read_rchar_bytes": None,
            "logical_read_rchar_exact_child_bytes": None,
            "rchar_method": "conservative_parent_self_io_delta_after_wait4",
            "descendant_scope": (
                "direct executable only; snapshot-detected descendants reject"
                " success; very short descendants can escape detection; no ha"
                "rd subtree bound"
            ),
        }
        self.commands.append(entry)
        save(journal, entry)  # Durable before launch, not only after a successful window.
        start = self.now()
        proc = token = before = usage = None
        observed = 0
        last_rchar = 0
        descendants = []
        census_failed = False
        waited = False

        def error(phase, exception):
            entry["errors"].append(
                {"phase": phase, "type": type(exception).__name__, "message": str(exception)}
            )

        def stop(reason):
            if entry["stop_reason"] is None:
                entry["stop_reason"] = reason

        def census():
            nonlocal census_failed, descendants
            try:
                members = owned_group_members(token)
            except BaseException:
                census_failed = True
                entry["descendant_scope_known"] = False
                entry["owned_group_after_cleanup"] = None
                entry["live_owned_group_after_cleanup"] = None
                raise
            entry["descendant_scope_known"] = not census_failed
            discovered = [item for item in members if item["pid"] != token["pid"]]
            if discovered:
                # Later samples cannot erase already observed, unwaited descendants.
                descendants = list(
                    {
                        (item["pid"], item["start_ticks"]): item
                        for item in descendants + discovered
                    }.values()
                )
                stop("unexpected_descendant_process")
            return members

        def ended():
            return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

        def signal_owned_group():
            confirm_owned_group(token)
            entry["termination_signal_attempted"] = "SIGKILL"
            os.killpg(proc.pid, signal.SIGKILL)
            entry["termination_signal_sent"] = "SIGKILL"

        def prepare_child():
            import resource

            os.sched_setaffinity(0, self.cores)
            resource.setrlimit(resource.RLIMIT_AS, (2 * GiB, 2 * GiB))

        try:
            before = read_io(os.getpid())
            with (
                prefix.with_suffix(".stdout").open("xb") as stdout,
                prefix.with_suffix(".stderr").open("xb") as stderr,
            ):
                proc = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=prepare_child,
                )
            entry["pid"] = proc.pid
            token = identity(proc.pid)
            confirm_owned_group(token)
            entry["owned_group_confirmed"] = True
            entry["owned_identity"] = token
            save(journal, entry)
            census()
            while True:
                wall = self.now()
                if wall >= clip_deadline:
                    stop("clip_wall_limit")
                    self.overshoots.append(
                        {"kind": "clip_wall_limit", "observed_overshoot": wall - clip_deadline}
                    )
                if wall - self.started >= self.total_seconds:
                    stop("total_wall_limit")
                if entry["stop_reason"] is not None or ended():
                    break
                try:
                    child = proc_stats(proc.pid)
                except (PermissionError, FileNotFoundError, ProcessExited) as exception:
                    # Only this child's kernel-confirmed exit race is reconciled.
                    if ended():
                        entry["exit_race_reconciled"] = type(exception).__name__
                        break
                    raise
                # The caller remains live even if its child just exited. Its monitoring
                # errors must never be reclassified using the child's wait state.
                parent = proc_stats(os.getpid())
                if child["rchar"] < last_rchar or parent["rchar"] < before["rchar"]:
                    raise GuardError("live rchar counter regressed")
                last_rchar = child["rchar"]
                observed = parent["rchar"] - before["rchar"] + child["rchar"]
                self.peak_rss = max(self.peak_rss, parent["rss"] + child["rss"])
                census()
                reasons = []
                if self.rchar + observed >= 2 * GiB:
                    reasons.append(("logical_read_limit", self.rchar + observed - 2 * GiB))
                if parent["rss"] + child["rss"] >= 4 * GiB:
                    reasons.append(("aggregate_rss_limit", parent["rss"] + child["rss"] - 4 * GiB))
                size = tree_bytes(self.root)
                if size >= 512 * MiB:
                    reasons.append(("output_tree_limit", size - 512 * MiB))
                free = shutil.disk_usage(self.root).free
                if free < GiB:
                    reasons.append(("disk_reserve_limit", GiB - free))
                if reasons:
                    stop(reasons[0][0])
                    self.overshoots.extend(
                        {"kind": kind, "observed_overshoot": value} for kind, value in reasons
                    )
                if entry["stop_reason"] is not None:
                    break
                time.sleep(self.poll_seconds)
        except BaseException as exception:
            stop("live_monitor_error" if proc is not None else "prelaunch_error")
            error("monitor" if proc is not None else "launch", exception)
        finally:
            if proc is not None:
                # Leader identity and the process census are deliberately independent.
                # A broken unrelated /proc entry cannot veto signaling our pinned group.
                identity_ok = False
                leader_ended = False
                try:
                    if token is None:
                        token = identity(proc.pid)
                    confirm_owned_group(token)
                    identity_ok = True
                    entry["owned_group_confirmed"] = True
                except BaseException as exception:
                    stop("cleanup_identity_error")
                    error("cleanup_identity", exception)
                try:
                    leader_ended = ended()
                except BaseException as exception:
                    stop("cleanup_wait_state_error")
                    error("cleanup_wait_state", exception)
                if identity_ok:
                    try:
                        census()
                        if descendants:
                            entry["leader_exited_before_descendant_cleanup"] = leader_ended
                    except BaseException as exception:
                        stop("cleanup_census_error")
                        error("cleanup_census", exception)
                    if entry["stop_reason"] is not None or descendants or census_failed:
                        try:
                            signal_owned_group()  # Same strict identity gate, no census.
                        except BaseException as exception:
                            stop("cleanup_error")
                            error("cleanup", exception)
                try:
                    cleanup_deadline = self.now() + 0.5
                    while True:
                        leader_ended = ended()
                        live = None
                        if identity_ok and not census_failed:
                            try:
                                members = census()
                                live = [item for item in members if item["state"] != "Z"]
                                entry["owned_group_after_cleanup"] = members
                                entry["live_owned_group_after_cleanup"] = live
                                if any(item["pid"] != token["pid"] for item in live):
                                    entry["leader_exited_before_descendant_cleanup"] = leader_ended
                                    try:
                                        signal_owned_group()
                                    except BaseException as signal_error:
                                        stop("cleanup_error")
                                        error("cleanup", signal_error)
                            except BaseException as exception:
                                stop("cleanup_census_error")
                                error("cleanup_census", exception)
                                if identity_ok and "termination_signal_attempted" not in entry:
                                    try:
                                        signal_owned_group()
                                    except BaseException as signal_error:
                                        error("cleanup", signal_error)
                        # Unknown census must not prevent wait4 of this confirmed child.
                        # It still forbids any complete descendant/accounting claim.
                        if leader_ended and (census_failed or not live):
                            break
                        if self.now() >= cleanup_deadline:
                            raise GuardError(
                                "owned leader/group not terminal after bounded cleanup; live "
                                "outcome unknown"
                            )
                        time.sleep(self.poll_seconds)
                    pid, status, usage = os.wait4(proc.pid, 0)
                    if pid != proc.pid:
                        raise GuardError("wait4 returned another process")
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    entry["returncode"] = proc.returncode
                    entry["reaped"] = waited = True
                    entry["wait4_rusage"] = {
                        "user_seconds": usage.ru_utime,
                        "system_seconds": usage.ru_stime,
                        "maxrss_bytes": usage.ru_maxrss * 1024,
                        "input_blocks": usage.ru_inblock,
                        "output_blocks": usage.ru_oublock,
                    }
                    self.cpu_seconds += usage.ru_utime + usage.ru_stime
                except BaseException as exception:
                    stop("wait_accounting_error")
                    error("wait", exception)
                if waited and before is not None:
                    try:
                        after = read_io(os.getpid())
                        charge = after["rchar"] - before["rchar"]
                        storage = after["read_bytes"] - before["read_bytes"]
                        if min(charge, storage) < 0 or charge < observed:
                            raise GuardError(
                                "final parent accounting regressed or omitted a live observed"
                                " charge"
                            )
                        entry["parent_self_io_before"] = before
                        entry["parent_self_io_after_wait4"] = after
                        entry["observed_conservative_final_rchar_charge_bytes"] = charge
                        entry["physical_read_bytes_observed"] = storage
                        if descendants or not entry["descendant_scope_known"]:
                            raise GuardError(
                                "descendant accounting is incomplete/unknown; direct-parent d"
                                "elta is not a subtree bound"
                            )
                        entry["logical_read_rchar_bytes"] = charge
                        entry["accounting_complete"] = True
                        self.rchar += charge
                        self.read_bytes += storage
                        if self.rchar >= 2 * GiB:
                            stop("logical_read_limit_final_accounting")
                            self.overshoots.append(
                                {
                                    "kind": "logical_read_limit_final_accounting",
                                    "observed_overshoot": self.rchar - 2 * GiB,
                                }
                            )
                    except BaseException as exception:
                        stop("final_accounting_error")
                        error("final_accounting", exception)
            if descendants:
                entry["unexpected_descendants"] = descendants
            entry["last_live_observed_conservative_rchar_bytes"] = observed
            entry["wall_seconds"] = self.now() - start
            self.child_runtime += entry["wall_seconds"] if proc is not None else 0
            entry["completed_utc"] = utc()
            if entry["returncode"] not in (None, 0):
                stop("child_exit_error")
            if not entry["accounting_complete"]:
                self.accounting_complete = False
                stop("incomplete_accounting")
            rejected = entry["stop_reason"] is not None or bool(entry["errors"])
            entry["status"] = "rejected" if rejected else "succeeded"
            self.failed |= rejected
            save(journal, entry)  # Cleanup/accounting exceptions cannot bypass terminal receipt.
        if self.failed:
            raise GuardError(
                f"child rejected: {prefix.name}: {entry['stop_reason']}; see {journal.name}"
            )
        return prefix.with_suffix(".stdout")
