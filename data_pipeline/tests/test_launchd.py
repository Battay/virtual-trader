"""Offline plist-generation tests that never install a LaunchAgent."""

from pathlib import Path
import plistlib

from data_pipeline.src.launchd import (
    LAUNCH_AGENT_LABEL,
    generate_launch_agent_plist,
)


def test_generates_expected_user_launch_agent_without_installing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "virtual-trader"
    python_path = project_root / ".venv" / "bin" / "python"
    logs_dir = project_root / "data_pipeline" / "logs"

    values = plistlib.loads(
        generate_launch_agent_plist(
            project_root=project_root,
            python_executable=python_path,
            logs_dir=logs_dir,
        )
    )

    assert values["Label"] == LAUNCH_AGENT_LABEL
    assert values["ProgramArguments"] == [
        str(python_path.absolute()),
        "-m",
        "data_pipeline.src.auto_update",
        "scheduled",
    ]
    assert values["WorkingDirectory"] == str(project_root.resolve())
    assert values["StartCalendarInterval"] == {"Hour": 17, "Minute": 15}
    assert values["RunAtLoad"] is False
    assert values["StandardOutPath"] == str(
        (logs_dir / "launchd.stdout.log").resolve()
    )
    assert values["StandardErrorPath"] == str(
        (logs_dir / "launchd.stderr.log").resolve()
    )
    assert not project_root.exists()
