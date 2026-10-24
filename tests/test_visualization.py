"""Unit tests for Trajecta visualization entry points and utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.visualization.viewer import (
    is_running_under_mjpython,
    launch_interactive_viewer,
    launch_passive_viewer,
    main,
)


class TestVisualizationModule:
    """Test suite verifying visualization interfaces without requiring an interactive window."""

    def test_is_running_under_mjpython_boolean(self) -> None:
        """Verify is_running_under_mjpython returns a boolean."""
        res = is_running_under_mjpython()
        assert isinstance(res, bool)

    def test_launch_interactive_viewer_reuses_provided_env_and_cleans_up(self) -> None:
        """Verify launch_interactive_viewer uses the provided env and calls env.close()."""
        env = MujocoRobotEnvironment()
        assert not env.is_closed

        with patch("mujoco.viewer.launch") as mock_launch:
            launch_interactive_viewer(env=env, show_left_ui=False, show_right_ui=False)
            mock_launch.assert_called_once_with(
                env.model,
                env.data,
                show_left_ui=False,
                show_right_ui=False,
            )

        assert env.is_closed

    def test_launch_interactive_viewer_creates_default_env_and_cleans_up(self) -> None:
        """Verify launch_interactive_viewer creates and closes default env if none provided."""
        closed_envs: list[MujocoRobotEnvironment] = []

        with patch("mujoco.viewer.launch") as mock_launch:
            with patch("roboforge.environments.mujoco_env.MujocoRobotEnvironment") as mock_env_cls:
                mock_env = MujocoRobotEnvironment()
                mock_env_cls.return_value = mock_env
                launch_interactive_viewer()
                mock_launch.assert_called_once_with(
                    mock_env.model,
                    mock_env.data,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                assert mock_env.is_closed

    def test_launch_passive_viewer_requires_mjpython_on_macos(self) -> None:
        """Verify launch_passive_viewer raises informative RuntimeError if not under mjpython on macOS."""
        with patch("sys.platform", "darwin"), patch("roboforge.visualization.viewer.is_running_under_mjpython", return_value=False):
            with pytest.raises(RuntimeError, match="mjpython"):
                launch_passive_viewer()

    def test_launch_passive_viewer_loop_execution_and_cleanup(self) -> None:
        """Verify passive viewer runs stepping loop and closes environment."""
        env = MujocoRobotEnvironment()

        # Mock viewer context manager with 2 iterations
        mock_viewer = MagicMock()
        mock_viewer.is_running.side_effect = [True, True, False]

        mock_launch_passive = MagicMock()
        mock_launch_passive.return_value.__enter__.return_value = mock_viewer

        with patch("sys.platform", "linux"), patch("mujoco.viewer.launch_passive", mock_launch_passive):
            launch_passive_viewer(env=env, max_duration=10.0)

        assert mock_viewer.sync.call_count == 2
        assert env.is_closed

    def test_cli_help(self) -> None:
        """Verify CLI entry point displays usage and flags with --help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_interactive_dispatch(self) -> None:
        """Verify CLI dispatches to launch_interactive_viewer by default."""
        with patch("roboforge.visualization.viewer.launch_interactive_viewer") as mock_interactive:
            ret = main([])
            assert ret == 0
            mock_interactive.assert_called_once_with(
                show_left_ui=True,
                show_right_ui=True,
            )

    def test_cli_passive_dispatch(self) -> None:
        """Verify CLI dispatches to launch_passive_viewer with --passive."""
        with patch("roboforge.visualization.viewer.launch_passive_viewer") as mock_passive:
            ret = main(["--passive", "--duration", "5.0", "--no-ui"])
            assert ret == 0
            mock_passive.assert_called_once_with(
                max_duration=5.0,
                show_left_ui=False,
                show_right_ui=False,
            )
