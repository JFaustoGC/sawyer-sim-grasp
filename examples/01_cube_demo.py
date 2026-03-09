"""
01_cube_demo.py — Pick and place a cube.

How to run:
    Windows:  python examples\01_cube_demo.py
    Linux:    python examples/01_cube_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sawyer_student import SawyerSim


def main():
    sim = SawyerSim(gui=True)

    CUBE_SIZE = 0.047           # metres — change this to resize the cube
    CUBE_X, CUBE_Y = 0.45, 0.13
    CUBE_Z = CUBE_SIZE / 2      # centre sits half a side above the floor

    sim.add_box(
        name="cube",
        position=[CUBE_X, CUBE_Y, CUBE_Z],
        size=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE],
        color=[0.9, 0.2, 0.2, 1.0],
    )

    sim.start()

    # ── Pick ──────────────────────────────────────────────────────────────────
    sim.open_gripper()
    sim.move_to(CUBE_X, CUBE_Y, CUBE_Z + 0.20)          # approach from above
    sim.move_to(CUBE_X, CUBE_Y, CUBE_Z + 0.015, speed="slow")  # descend
    sim.close_gripper()
    sim.wait(1.0)

    # ── Place ─────────────────────────────────────────────────────────────────
    sim.move_to(CUBE_X, CUBE_Y, CUBE_Z + 0.20)          # lift
    sim.move_to(CUBE_X, CUBE_Y - 0.25, CUBE_Z + 0.20)   # move sideways
    sim.move_to(CUBE_X, CUBE_Y - 0.25, CUBE_Z + 0.015, speed="slow")  # set down
    sim.open_gripper()
    sim.wait(0.5)

    sim.reset()
    sim.wait(1.0)
    sim.close()


if __name__ == "__main__":
    main()
