# Sawyer Student Lab

A beginner-friendly simulation environment for the Sawyer robot arm.

---

## Installation (one time only)

1. Install **Python 3.10** from https://www.python.org/downloads/
   During install: check **"Add Python to PATH"**.

2. Open a terminal (Windows: press `Win+R`, type `cmd`, press Enter) and run:

```
pip install mujoco numpy scipy casadi
```

That's it. You do **not** need to install any `sawyer_*` packages — they are
bundled inside the `_sawyer/` folder.

---

## How to run the examples (Windows)

Open a terminal in this folder and run:

```
python examples\01_cube_demo.py
python examples\02_cylinder_urdf_demo.py
```

A 3-D viewer will open. The robot picks up the object and places it to the side.

---

## How to run the notebooks

```
pip install jupyter
jupyter notebook
```

Then open the `notebooks/` folder in the browser tab that appears.

---

## Folder structure

```
sawyer-student-lab/
├── _sawyer/                    ← bundled robot libraries (do not edit)
├── sawyer_student/
│   └── sim.py                  ← SawyerSim class (the whole API)
├── notebooks/
│   ├── 01_getting_started.ipynb
│   ├── 02_pick_and_place.ipynb
│   └── 03_custom_objects.ipynb
├── examples/
│   ├── 01_cube_demo.py         ← pick and place a box
│   ├── 02_cylinder_urdf_demo.py← load object from URDF file
│   └── objects/
│       └── cylinder.urdf       ← put your own URDF files here
└── README.md
```

---

## Quick API reference

```python
from sawyer_student import SawyerSim

sim = SawyerSim(gui=True)

# Add objects BEFORE start()
sim.add_box("cube", position=[0.45, 0.0, 0.028], size=[0.057, 0.057, 0.057])
sim.add_sphere("ball", position=[0.45, 0.2, 0.03], radius=0.03)
sim.add_cylinder("can", position=[0.45, -0.2, 0.05], radius=0.03, height=0.10)
sim.add_object_from_urdf("my_object.urdf", position=[0.4, 0.0, 0.03])

sim.start()                          # opens viewer, moves to ready pose

sim.move_to(0.45, 0.0, 0.20)        # move gripper (world frame, metres)
sim.open_gripper()
sim.close_gripper()
x, y, z = sim.get_position()        # read gripper position
sim.wait(1.0)                        # pause N seconds
sim.reset()                          # return to ready pose
sim.close()                          # shut down
```

### Cube size reference

The standard demo cube is **5.7 cm × 5.7 cm × 5.7 cm** (`size=[0.057, 0.057, 0.057]`).
Place it with `z = 0.057/2 = 0.0285` so it sits on the floor.
# sawyer-sim-grasp
