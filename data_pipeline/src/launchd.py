"""Generate and manage the user-level macOS LaunchAgent for PSX updates."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import plistlib
import platform
import subprocess
from typing import Sequence

from .config import LOGS_DIR, PROJECT_ROOT


LAUNCH_AGENT_LABEL = "com.virtual-trader.psx-auto-update"
LAUNCH_HOUR = 17
LAUNCH_MINUTE = 15


@dataclass(frozen=True)
class LaunchAgentStatus:
    """Current installation and launchd loading state."""

    plist_path: Path
    installed: bool
    loaded: bool
    detail: str


def default_python_executable(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the repository virtual-environment Python executable."""
    return Path(project_root) / ".venv" / "bin" / "python"


def default_launch_agent_path() -> Path:
    """Return the current user's standard LaunchAgents plist path."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def launch_domain() -> str:
    """Return the launchd GUI domain for the current non-root user."""
    return f"gui/{os.getuid()}"


def build_launch_agent(
    *,
    project_root: Path = PROJECT_ROOT,
    python_executable: Path | None = None,
    logs_dir: Path = LOGS_DIR,
) -> dict[str, object]:
    """Build a portable LaunchAgent definition without installing it."""
    root = Path(project_root).resolve()
    python_path = (
        Path(python_executable).absolute()
        if python_executable is not None
        else default_python_executable(root).absolute()
    )
    log_directory = Path(logs_dir).resolve()
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            str(python_path),
            "-m",
            "data_pipeline.src.auto_update",
            "scheduled",
        ],
        "WorkingDirectory": str(root),
        "StartCalendarInterval": {
            "Hour": LAUNCH_HOUR,
            "Minute": LAUNCH_MINUTE,
        },
        "StandardOutPath": str(log_directory / "launchd.stdout.log"),
        "StandardErrorPath": str(log_directory / "launchd.stderr.log"),
        "RunAtLoad": False,
    }


def generate_launch_agent_plist(
    *,
    project_root: Path = PROJECT_ROOT,
    python_executable: Path | None = None,
    logs_dir: Path = LOGS_DIR,
) -> bytes:
    """Serialize the LaunchAgent definition as a valid XML plist."""
    return plistlib.dumps(
        build_launch_agent(
            project_root=project_root,
            python_executable=python_executable,
            logs_dir=logs_dir,
        ),
        fmt=plistlib.FMT_XML,
        sort_keys=False,
    )


def _require_macos() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("LaunchAgent management is available only on macOS")


def _run_launchctl(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/launchctl", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def write_launch_agent_plist(
    path: Path = default_launch_agent_path(),
    *,
    project_root: Path = PROJECT_ROOT,
    python_executable: Path | None = None,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Write a LaunchAgent plist after validating its local executable paths."""
    root = Path(project_root).resolve()
    python_path = (
        Path(python_executable).absolute()
        if python_executable is not None
        else default_python_executable(root).absolute()
    )
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Virtual-environment Python was not found at {python_path}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    content = generate_launch_agent_plist(
        project_root=root,
        python_executable=python_path,
        logs_dir=logs_dir,
    )
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def get_launch_agent_status(
    path: Path = default_launch_agent_path(),
) -> LaunchAgentStatus:
    """Inspect whether the plist exists and whether launchd has loaded it."""
    plist_path = Path(path)
    if platform.system() != "Darwin":
        return LaunchAgentStatus(
            plist_path=plist_path,
            installed=plist_path.is_file(),
            loaded=False,
            detail="LaunchAgent status is available only on macOS",
        )

    try:
        result = _run_launchctl(
            ["print", f"{launch_domain()}/{LAUNCH_AGENT_LABEL}"]
        )
    except OSError as exc:
        return LaunchAgentStatus(
            plist_path=plist_path,
            installed=plist_path.is_file(),
            loaded=False,
            detail=f"Could not inspect launchd: {exc}",
        )
    loaded = result.returncode == 0
    detail = (
        "LaunchAgent is loaded"
        if loaded
        else (result.stderr.strip() or "LaunchAgent is not loaded")
    )
    return LaunchAgentStatus(
        plist_path=plist_path,
        installed=plist_path.is_file(),
        loaded=loaded,
        detail=detail,
    )


def install_launch_agent(
    path: Path = default_launch_agent_path(),
) -> LaunchAgentStatus:
    """Explicitly write and bootstrap the user LaunchAgent."""
    _require_macos()
    plist_path = write_launch_agent_plist(path)
    target = f"{launch_domain()}/{LAUNCH_AGENT_LABEL}"
    if _run_launchctl(["print", target]).returncode == 0:
        bootout = _run_launchctl(["bootout", target])
        if bootout.returncode != 0:
            raise RuntimeError(bootout.stderr.strip() or "Could not reload LaunchAgent")

    result = _run_launchctl(["bootstrap", launch_domain(), str(plist_path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not install LaunchAgent")
    return get_launch_agent_status(plist_path)


def uninstall_launch_agent(
    path: Path = default_launch_agent_path(),
) -> LaunchAgentStatus:
    """Explicitly unload and remove the user LaunchAgent plist."""
    _require_macos()
    plist_path = Path(path)
    target = f"{launch_domain()}/{LAUNCH_AGENT_LABEL}"
    if _run_launchctl(["print", target]).returncode == 0:
        result = _run_launchctl(["bootout", target])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Could not unload LaunchAgent")
    try:
        plist_path.unlink()
    except FileNotFoundError:
        pass
    return get_launch_agent_status(plist_path)


def trigger_launch_agent() -> None:
    """Ask launchd to run the already-installed updater immediately."""
    _require_macos()
    target = f"{launch_domain()}/{LAUNCH_AGENT_LABEL}"
    result = _run_launchctl(["kickstart", "-k", target])
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Could not trigger the installed LaunchAgent"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Manage the user-level PSX automatic-update LaunchAgent."""
    parser = argparse.ArgumentParser(description="Manage PSX macOS automation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Generate the user plist without loading it")
    subparsers.add_parser("install", help="Install and load the user LaunchAgent")
    subparsers.add_parser("status", help="Inspect LaunchAgent installation status")
    subparsers.add_parser("trigger", help="Trigger the installed LaunchAgent now")
    subparsers.add_parser("uninstall", help="Unload and remove the LaunchAgent")
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            path = write_launch_agent_plist()
            print(f"Generated LaunchAgent: {path}")
        elif args.command == "install":
            status = install_launch_agent()
            print(f"Installed LaunchAgent: {status.plist_path}")
            print(status.detail)
        elif args.command == "status":
            status = get_launch_agent_status()
            print(f"Plist installed: {'yes' if status.installed else 'no'}")
            print(f"Loaded by launchd: {'yes' if status.loaded else 'no'}")
            print(f"Plist path: {status.plist_path}")
            print(status.detail)
        elif args.command == "trigger":
            trigger_launch_agent()
            print("LaunchAgent trigger requested")
        else:
            status = uninstall_launch_agent()
            print(f"Removed LaunchAgent: {status.plist_path}")
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
