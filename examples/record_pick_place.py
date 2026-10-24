#!/usr/bin/env python3
"""Render a clean MP4 of the verified Panda pick-and-place simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from roboforge.controllers.end_effector_position import PandaEndEffectorPositionController
from roboforge.environments.mujoco_env import MujocoRobotEnvironment
from roboforge.tasks.pick_place import PickPlaceConfig, PickPlaceTask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("media/trajecta_pick_and_place_1080p.mp4"))
    parser.add_argument("--frames-dir", type=Path, default=Path("media/frames"))
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps, width, and height must be positive")

    env = MujocoRobotEnvironment(max_episode_steps=10_000)
    task = PickPlaceTask(env, PandaEndEffectorPositionController(env, position_tolerance=0.005), PickPlaceConfig())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        env.close()
        raise RuntimeError(f"Could not open MP4 writer for {args.output}; check OpenCV video codec support")

    renderer = None
    try:
        task.reset(seed=args.seed)
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), args.width)
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), args.height)
        env.model.vis.headlight.ambient[:] = (0.34, 0.34, 0.34)
        env.model.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
        env.model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (0.38, 0.0, 0.87)
        camera.distance = 1.8
        camera.azimuth = 135
        camera.elevation = -24
        backdrop = np.broadcast_to(
            np.linspace((22, 29, 38), (48, 59, 69), args.height, dtype=np.uint8)[:, None, :],
            (args.height, args.width, 3),
        ).copy()
        saved_phases: set[str] = set()

        def write_frame(label: str, *, success: bool | None = None) -> None:
            renderer.update_scene(env.data, camera=camera)
            rgb = renderer.render()
            # Give the otherwise black offscreen backdrop a restrained deep-slate gradient.
            dark = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 0, 3)
            cv2.copyTo(backdrop, dark, rgb)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Subtle legible title strip for a polished, self-contained clip.
            cv2.rectangle(frame, (0, 0), (args.width, 84), (20, 25, 31), -1)
            cv2.putText(frame, "TRAJECTA", (44, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (225, 238, 244), 2, cv2.LINE_AA)
            cv2.putText(frame, "FRANKA PANDA  |  MUJOCO SIMULATION", (44, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (173, 190, 201), 1, cv2.LINE_AA)
            status = label if success is None else ("PICK AND PLACE COMPLETE" if success else "TASK FAILED")
            color = (85, 220, 150) if success else (245, 206, 115)
            text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
            cv2.putText(frame, status, (args.width - text_size[0] - 44, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            writer.write(frame)
            if label in {"DESCEND", "VERIFY_GRASP", "LIFT", "TRANSPORT", "VERIFY_PLACEMENT"} and label not in saved_phases:
                args.frames_dir.mkdir(parents=True, exist_ok=True)
                filename = label.lower() + ".jpg"
                cv2.imwrite(str(args.frames_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
                saved_phases.add(label)
            if success is True and "SUCCESS" not in saved_phases:
                args.frames_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.frames_dir / "success.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
                saved_phases.add("SUCCESS")

        # Short opening slate, then one frame per simulation control step.
        for _ in range(args.fps * 2):
            write_frame("CONTACT-VERIFIED PICK AND PLACE")
        while task.result is None:
            result = task.step()
            write_frame(task.phase.value.replace("_", " "))
            if result.terminated or result.truncated:
                break
        final = task.result
        for _ in range(args.fps * 3):
            write_frame("", success=bool(final and final.success))
        if final is None or not final.success:
            reason = final.failure_reason if final else "task did not return a result"
            raise RuntimeError(f"Pick-and-place did not succeed; video retained for inspection. {reason}")
        print(f"Video: {args.output.resolve()}")
        print(f"Frames: {args.frames_dir.resolve()}")
        print(f"Success: {final.success}; seed={args.seed}; steps={final.episode_length}; placement_error_m={final.final_placement_error_m:.5f}")
        print(f"Resolution: {args.width}x{args.height}; FPS: {args.fps}; duration approximately {2 + final.episode_length / args.fps + 3:.1f}s")
        return 0
    finally:
        if renderer is not None:
            renderer.close()
        writer.release()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
