# Sawyer Student Lab

A beginner-friendly simulation environment for the Sawyer robot arm.

---

## Installation (one time only — WSL + Conda)

### 1. Open WSL (Ubuntu on Windows)
Press `Win+R`, type `wsl`, press Enter.

### 2. Install Miniconda (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Restart the terminal after installation.

### 3. Clone or download this repository, then create the environment
```bash
cd sawyer-sim-grasp          # or wherever you put the folder
conda env create -f environment.yml
conda activate sawyer-sim
```

That's it. You do **not** need to install any `sawyer_*` packages — they are
bundled inside the `_sawyer/` folder.

---

## How to run the examples

```bash
conda activate sawyer-sim
python examples/01_cube_demo.py
python examples/02_cylinder_urdf_demo.py
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

# Joint control
joints = sim.get_joints()            # read 7 joint angles [j0…j6] in radians
joints[6] += 0.5                     # rotate wrist ~30° (changes gripper orientation)
sim.set_joints(joints)               # teleport instantly (no animation)
sim.move_joints(joints)              # move smoothly with planned trajectory
# Ready pose joint angles: [0.0, -0.9, 0.0, 1.8, 0.0, 0.6, 1.776047]
# Last value (j6 = 1.776047) sets the gripper facing forward — change it to rotate.
```

### Cube size reference

The standard demo cube is **5.7 cm × 5.7 cm × 5.7 cm** (`size=[0.057, 0.057, 0.057]`).
Place it with `z = 0.057/2 = 0.0285` so it sits on the floor.
# sawyer-sim-grasp
