"""Egress-attempt recorder, injected by the egress harness via PYTHONPATH.

Registers a ``sys.addaudithook`` (PEP 578) that appends one JSON line per *non-loopback*
network attempt made by any Python process in the test (socket.connect / connect_ex /
sendto / sendmsg / getaddrinfo / gethostbyname) to ``$ZLLF_EGRESS_AUDIT_LOG``.

It never blocks anything: the OS layer (network namespace / --network none / firewall) does
that. Its job is to catch attempts that fail *silently* (a library that swallows ENETUNREACH)
so the test can still fail. Only host/port and the event name are recorded, never payloads.
Inactive unless ZLLF_EGRESS_AUDIT_LOG is set. Chains to any other sitecustomize on sys.path.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

_LOG = os.environ.get("ZLLF_EGRESS_AUDIT_LOG")
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", ""}
_EVENTS = {
    "socket.connect",
    "socket.sendto",
    "socket.sendmsg",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.gethostbyaddr",
}


def _host_of(event: str, args: tuple[object, ...]) -> object:
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        addr = args[1] if len(args) > 1 else None
        return addr[0] if isinstance(addr, tuple) and addr else addr
    return args[0] if args else None


def _is_loopback(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):  # AF_UNIX paths etc.
        return True
    h = host.strip("[]").lower()
    return (
        h in _LOOPBACK_NAMES
        or h.startswith("127.")
        or h in ("::1", "0:0:0:0:0:0:0:1", "0.0.0.0", "::")
        or h.startswith("/")  # unix socket path
        or h.startswith("\x00")  # abstract unix socket
    )


def _hook(event: str, args: tuple[object, ...]) -> None:
    if event not in _EVENTS:
        return
    try:
        if (
            event == "socket.connect"
            and args
            and getattr(args[0], "family", None) not in (2, 10, 30)
        ):
            return  # only AF_INET / AF_INET6 (10 on Linux, 30 on macOS)
        host = _host_of(event, args)
        if _is_loopback(host):
            return
        port = None
        if event == "socket.getaddrinfo" and len(args) > 1:
            port = args[1]
        elif event.startswith("socket.") and len(args) > 1 and isinstance(args[1], tuple):
            port = args[1][1] if len(args[1]) > 1 else None
        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "event": event,
            "host": str(host),
            "port": None if port is None else str(port),
            "argv0": sys.argv[0] if sys.argv else "",
        }
        with open(_LOG, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # never break the program under test
        pass


if _LOG:
    sys.addaudithook(_hook)

# Chain to a sitecustomize that ours shadows (e.g. a distro one), if any.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if _p and os.path.abspath(_p) != _here and os.path.isfile(os.path.join(_p, "sitecustomize.py")):
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "_zllf_chained_sitecustomize", os.path.join(_p, "sitecustomize.py")
        )
        if _spec and _spec.loader:
            with contextlib.suppress(Exception):
                _spec.loader.exec_module(_ilu.module_from_spec(_spec))
        break
