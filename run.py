"""Main entry for Boeing 747 circular-flight simulation with LQR control."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B747 circular-flight simulation")
    parser.add_argument(
        "--show-animation",
        action="store_true",
        default=True,
        help="Play point-cloud animation in an interactive window.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Animation frame rate.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames used for animation playback/export (0 means use all simulation frames).",
    )
    return parser


import matplotlib


import matplotlib.pyplot as plt

from controller import CircleRef, ControlConfig, LQRCircleController
from dynamics import B747Params, rk4_step
from visual import save_flight_animation, show_flight_animation


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def simulate():
    p = B747Params()
    cfg = ControlConfig(v_ref=210.0, dt=0.05)
    ref = CircleRef(center_x=0.0, center_y=0.0, radius=12_0000.0, ccw=True)

    ctrl = LQRCircleController(cfg, p)

    t_end = 800.0
    n = int(t_end / cfg.dt)
    time = np.arange(n + 1) * cfg.dt

    # Start outside boundary with heading slightly off tangent.
    state = np.array([ref.radius + 2000.0, 0.0, np.deg2rad(96.0), 0.0, cfg.v_ref], dtype=float)

    hist = np.zeros((n + 1, 5), dtype=float)
    e_ct_hist = np.zeros(n + 1, dtype=float)
    e_psi_hist = np.zeros(n + 1, dtype=float)
    phi_cmd_hist = np.zeros(n + 1, dtype=float)

    hist[0] = state

    for k in range(n):
        control, info = ctrl.compute_control(state, ref)
        state = rk4_step(state, control, cfg.dt, p)

        hist[k + 1] = state
        e_ct_hist[k + 1] = info["e_ct"]
        e_psi_hist[k + 1] = np.rad2deg(info["e_psi"])
        phi_cmd_hist[k + 1] = np.rad2deg(control["phi_cmd"])

    return time, hist, e_ct_hist, e_psi_hist, phi_cmd_hist, ref


def plot_results(time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref):
    x = hist[:, 0]
    y = hist[:, 1]
    phi_deg = np.rad2deg(hist[:, 3])
    v = hist[:, 4]

    th = np.linspace(0.0, 2.0 * np.pi, 600)
    x_ref = ref.center_x + ref.radius * np.cos(th)
    y_ref = ref.center_y + ref.radius * np.sin(th)

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_ref, y_ref, "k--", linewidth=1.5, label="Reference Circle")
    ax1.plot(x, y, "b", linewidth=1.5, label="B747 Trajectory")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Circular Flight Tracking")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, e_ct, "r")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("signed radius error (m)")
    ax2.set_title("Cross-Track Error (positive=inside)")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time, phi_deg, label="phi")
    ax3.plot(time, phi_cmd_deg, "--", label="phi_cmd")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("bank angle (deg)")
    ax3.set_title("Bank Response")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, v, label="v")
    ax4.plot(time, e_psi_deg, label="heading err (deg)")
    ax4.set_xlabel("time (s)")
    ax4.set_title("Speed and Heading Error")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    os.makedirs("data", exist_ok=True)
    fig.savefig("data/circle_flight_result.png", dpi=150)


def main():
    args = parse_args()

    time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref = simulate()
    print(f"Final position: x={hist[-1,0]:.1f} m, y={hist[-1,1]:.1f} m")
    print(f"Final speed: {hist[-1,4]:.2f} m/s")
    print(f"Final signed cross-track error: {e_ct[-1]:.2f} m")
    print(f"Final |cross-track error|: {abs(e_ct[-1]):.2f} m")

    # plot_results(time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref)
    # print("Saved figure: data/circle_flight_result.png")

    print("Opening animation window...")
    anim = show_flight_animation(
        time=time,
        hist=hist,
        ref=ref,
        fps=args.fps,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
