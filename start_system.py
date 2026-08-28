#!/usr/bin/env python3
"""
Startup script for Medical Evidence Graph & Outcomes Insight Lab.
Launches the API backend and the Streamlit frontend together.

Binds to loopback by default. Exposing a clinical analysis API on every
interface is a decision, not a default, and `--host` is how you make it:
the previous version passed 0.0.0.0 to both services unconditionally, so
the one-command launcher published an unauthenticated patient-analysis API
to the whole network. Binding beyond loopback now requires API keys to be
configured, and refuses without them.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests


LOOPBACK = "127.0.0.1"
REQUIRED_FILES = ("src/api_backend.py", "src/frontend_interface.py")


def port_is_free(port: int, host: str = LOOPBACK) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def wait_until_healthy(url: str, process: subprocess.Popen,
                       timeout: float = 90.0) -> bool:
    """Poll until the service answers, or it exits, or we run out of time.

    Replaces a fixed `time.sleep(3)` followed by one attempt. The API loads
    scikit-learn and torch at import, which takes longer than three seconds
    on a cold start, so that check reported failure for a service that was
    merely still starting.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def start_api_server(host: str, port: int, reload: bool) -> Optional[subprocess.Popen]:
    print(f"Starting API backend on {host}:{port} ...")
    command = [
        sys.executable, "-m", "uvicorn", "src.api_backend:app",
        "--host", host, "--port", str(port),
    ]
    if reload:
        command.append("--reload")

    # Output is inherited rather than piped. A piped stream nobody reads
    # fills its buffer and blocks the child, which for uvicorn under load
    # means the API stops serving with no indication why.
    process = subprocess.Popen(command, cwd=os.getcwd())

    if wait_until_healthy(f"http://{LOOPBACK}:{port}/api/health", process):
        print("API backend is up.")
        return process

    print("API backend did not become healthy.")
    process.terminate()
    return None


def start_frontend(host: str, port: int, api_url: str) -> Optional[subprocess.Popen]:
    print(f"Starting frontend on {host}:{port} ...")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            # A file path. Streamlit rejects a module:attribute target
            # outright ("Streamlit requires raw Python (.py) files"), so
            # the previous `src.frontend_interface:main` could never have
            # started -- and the error handler reported it as "normal
            # during initial load" and returned the process as a success.
            "src/frontend_interface.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
        ],
        cwd=os.getcwd(),
        env={**os.environ, "MEG_API_URL": api_url},
    )

    if wait_until_healthy(f"http://{LOOPBACK}:{port}/_stcore/health", process):
        print("Frontend is up.")
        return process

    print("Frontend did not become healthy.")
    process.terminate()
    return None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the API and the frontend")
    parser.add_argument(
        "--host", default=LOOPBACK,
        help=f"interface to bind (default {LOOPBACK}; anything else needs API keys)")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--reload", action="store_true", help="uvicorn auto-reload")
    args = parser.parse_args(argv)

    print("=" * 62)
    print("MEDICAL EVIDENCE GRAPH & OUTCOMES INSIGHT LAB")
    print("=" * 62)

    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print(f"Required files not found: {missing}")
        return 1

    # The check that matters. This service trains risk models and analyses
    # patient cohorts; publishing it unauthenticated is not something a
    # convenience script should be able to do by accident.
    if args.host != LOOPBACK:
        from src.api_backend import load_api_keys

        if not load_api_keys():
            print(
                f"Refusing to bind {args.host} with no API keys configured.\n"
                f"Anyone who can reach that interface could train models and "
                f"read the corpus.\n"
                f"Set MEG_API_KEYS, or drop --host to stay on {LOOPBACK}.")
            return 1
        print(f"Binding {args.host} with API-key authentication enabled.")

    for label, port in (("API", args.api_port), ("frontend", args.frontend_port)):
        if not port_is_free(port):
            print(f"Port {port} ({label}) is already in use.")
            return 1

    api_process = start_api_server(args.host, args.api_port, args.reload)
    if not api_process:
        return 1

    api_url = f"http://{LOOPBACK}:{args.api_port}"
    frontend_process = start_frontend(args.host, args.frontend_port, api_url)
    if not frontend_process:
        print("API is running; start the frontend yourself with:")
        print("  streamlit run src/frontend_interface.py")

    print("\n" + "=" * 62)
    print(f"API docs:     http://{LOOPBACK}:{args.api_port}/docs")
    print(f"Health:       http://{LOOPBACK}:{args.api_port}/api/health")
    if frontend_process:
        print(f"Frontend:     http://{LOOPBACK}:{args.frontend_port}")
    print("=" * 62)
    print("\nCtrl+C to shut down.\n")

    try:
        while True:
            time.sleep(1)
            if api_process.poll() is not None:
                print("API backend exited.")
                break
            if frontend_process and frontend_process.poll() is not None:
                print("Frontend exited.")
                break
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        for process in (frontend_process, api_process):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
