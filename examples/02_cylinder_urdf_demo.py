"""
02_cylinder_urdf_demo.py — Load a custom object from a URDF file and pick it.

The URDF is in examples/objects/cylinder.urdf — open it to see how it is written.
Put your own URDF files in that same folder and swap the path below.

How to run:
    Windows:  python examples\02_cylinder_urdf_demo.py
    Linux:    python examples/02_cylinder_urdf_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sawyer_student import SawyerSim

# Path to the URDF — relative to this script
URDF_PATH = os.path.join(os.path.dirname(__file__), "objects", "cylinder.urdf")

# Cylinder dimensions (must match what is written in the URDF)
CYLINDER_RADIUS = 0.0235   # metres  (47 mm diameter)
CYLINDER_HEIGHT = 0.080    # metres  (80 mm tall)


def main():
    sim = SawyerSim(gui=True)

    CYL_X, CYL_Y = 0.45, 0.13
    CYL_Z = CYLINDER_HEIGHT / 2   # centre is half the height above the floor

    sim.add_object_from_urdf(
        urdf_path=URDF_PATH,
        position=[CYL_X, CYL_Y, CYL_Z],
        name="cylinder",
    )

    sim.start()

    # ── Pick ──────────────────────────────────────────────────────────────────
    sim.open_gripper()
    sim.move_to(CYL_X, CYL_Y, CYL_Z + 0.20)          # approach from above
    sim.move_to(CYL_X, CYL_Y, CYL_Z + 0.015, speed="slow")  # descend
    sim.close_gripper()
    sim.wait(1.0)

    # ── Place ─────────────────────────────────────────────────────────────────
    sim.move_to(CYL_X, CYL_Y, CYL_Z + 0.20)          # lift
    sim.move_to(CYL_X, CYL_Y - 0.25, CYL_Z + 0.20)   # move sideways
    sim.move_to(CYL_X, CYL_Y - 0.25, CYL_Z + 0.015, speed="slow")  # set down
    sim.open_gripper()
    sim.wait(0.5)

    sim.reset()
    sim.wait(1.0)
    sim.close()


if __name__ == "__main__":
    main()
