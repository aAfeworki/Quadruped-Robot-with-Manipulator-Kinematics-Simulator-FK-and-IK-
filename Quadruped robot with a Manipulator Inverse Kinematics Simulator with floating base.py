import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# PARAMETERS
# -----------------------------
L1, L2 = 0.2, 0.2
Lm1, Lm2, Lm3 = 0.02, 0.2, 0.2

# Base pose
base_pose = [0, 0, 0, 0, 0, 0]
base_defaults = base_pose[:]

# Arm target (WORLD FRAME)
arm_target = np.array([0.4, 0.0, 0.1])
arm_default = arm_target.copy()

# Initializing theta3 (index 2) to -1.0 to force "Elbow Up" at start
prev_arm_q = np.array([0.0, 0.0, -1.0])

# Fixed feet
feet_world = {
    "FR": np.array([0.3, -0.2, -0.3]),
    "FL": np.array([0.3, 0.2, -0.3]),
    "RR": np.array([-0.3, -0.2, -0.3]),
    "RL": np.array([-0.3, 0.2, -0.3])
}

hip_offsets = {
    "FR": np.array([0.3, -0.2, 0]),
    "FL": np.array([0.3, 0.2, 0]),
    "RR": np.array([-0.3, -0.2, 0]),
    "RL": np.array([-0.3, 0.2, 0])
}

arm_offset = np.array([0.24, 0, 0])

body = np.array([
    [0.3, -0.2, 0], [0.3, 0.2, 0],
    [-0.3, 0.2, 0], [-0.3, -0.2, 0], [0.3, -0.2, 0]
])


# -----------------------------
# ROTATION
# -----------------------------
def rot_matrix(r, p, y):
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# -----------------------------
# IK LEG
# -----------------------------
def ik_leg(x, y, z):
    t1 = np.arctan2(y, -z)
    R = max(np.sqrt(y ** 2 + z ** 2), 1e-6)

    D = x ** 2 + R ** 2
    c = np.clip((D - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
    t3 = np.arccos(c)

    t2 = np.arctan2(x, R) - np.arctan2(L2 * np.sin(t3), L1 + L2 * np.cos(t3))
    return np.array([t1, t2, t3])


# -----------------------------
# IK ARM (AUTO ELBOW)
# -----------------------------
def ik_arm(X, Y, Z):
    global prev_arm_q

    t1 = np.arctan2(Y, X)

    r = np.sqrt(X ** 2 + Y ** 2)
    z = Z - Lm1
    D = np.sqrt(r ** 2 + z ** 2)

    c = np.clip((D ** 2 - Lm2 ** 2 - Lm3 ** 2) / (2 * Lm2 * Lm3), -1, 1)

    t3a = np.arccos(c)
    t3b = -np.arccos(c)

    def solve(t3):
        t2 = np.arctan2(z, r) - np.arctan2(Lm3 * np.sin(t3), Lm2 + Lm3 * np.cos(t3))
        return np.array([t1, t2, t3])

    qa = solve(t3a)
    qb = solve(t3b)

    # pick closest to previous (smooth)
    if np.linalg.norm(qa - prev_arm_q) < np.linalg.norm(qb - prev_arm_q):
        q = qa
    else:
        q = qb

    prev_arm_q = q
    return q


# -----------------------------
# FK
# -----------------------------
def fk_leg(t1, t2, t3):
    X = L1 * np.sin(t2) + L2 * np.sin(t2 + t3)
    R = L1 * np.cos(t2) + L2 * np.cos(t2 + t3)
    return np.array([X, R * np.sin(t1), -R * np.cos(t1)])


def fk_arm(t1, t2, t3):
    X = (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.cos(t1)
    Y = (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.sin(t1)
    Z = Lm2 * np.sin(t2) + Lm3 * np.sin(t2 + t3) + Lm1
    return np.array([X, Y, Z])


# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


def draw():
    ax.clear()

    bx, by, bz, r, p, y = base_pose
    Rb = rot_matrix(r, p, y)
    base = np.array([bx, by, bz])

    # Body
    body_w = (Rb @ body.T).T + base
    ax.plot(body_w[:, 0], body_w[:, 1], body_w[:, 2], linewidth=3)

    angles = {}

    # ---- LEGS ----
    for leg in feet_world:
        foot_w = feet_world[leg]
        hip_w = base + Rb @ hip_offsets[leg]

        foot_local = Rb.T @ (foot_w - hip_w)

        q = ik_leg(*foot_local)
        angles[leg] = np.degrees(q)

        knee = hip_w + Rb @ np.array([
            L1 * np.sin(q[1]),
            L1 * np.cos(q[1]) * np.sin(q[0]),
            -L1 * np.cos(q[1]) * np.cos(q[0])
        ])

        foot_calc = hip_w + Rb @ fk_leg(*q)

        ax.plot(*zip(hip_w, knee), marker='o')
        ax.plot(*zip(knee, foot_calc), marker='o')
        ax.scatter(*foot_w, s=25)

    # ---- ARM ----
    arm_base_w = base + Rb @ arm_offset

    arm_local = Rb.T @ (arm_target - arm_base_w)

    q = ik_arm(*arm_local)
    angles["ARM"] = np.degrees(q)

    j1 = arm_base_w + Rb @ [0, 0, Lm1]
    j2 = j1 + Rb @ [
        Lm2 * np.cos(q[1]) * np.cos(q[0]),
        Lm2 * np.cos(q[1]) * np.sin(q[0]),
        Lm2 * np.sin(q[1])
    ]
    end = arm_base_w + Rb @ fk_arm(*q)

    ax.plot(*zip(arm_base_w, j1), marker='o')
    ax.plot(*zip(j1, j2), marker='o')
    ax.plot(*zip(j2, end), marker='o')
    ax.scatter(*arm_target, s=40, c='r')

    # ---- INFO ----
    txt = "         θ1     θ2     θ3\n"
    for k in ["FR", "FL", "RR", "RL", "ARM"]:
        a = angles[k]
        txt += f"{k}: {a[0]:6.1f} {a[1]:6.1f} {a[2]:6.1f}\n"

    ax.text2D(0.02, 0.98, txt, transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8),
              va='top')

    ax.set(xlim=[-0.8, 0.8], ylim=[-0.8, 0.8], zlim=[-0.8, 0.5])


# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Floating Base + Fixed Legs + Arm IK")
root.geometry("1200x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel = tk.Frame(root)
panel.pack(side=tk.RIGHT, fill=tk.Y)

labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
limits = [(-0.27, 0.27), (-0.27, 0.27), (-0.3, 0.1),
          (-0.56, 0.56), (-0.35, 0.35), (-0.6, 0.6)]

sliders = []


def update(i, v):
    base_pose[i] = float(v)
    draw();
    canvas.draw_idle()


frame = tk.LabelFrame(panel, text="Base Control")
frame.pack(fill="x")

for i, n in enumerate(labels):
    s = tk.Scale(frame, from_=limits[i][0], to=limits[i][1],
                 resolution=0.01, orient=tk.HORIZONTAL,
                 label=n, command=lambda v, j=i: update(j, v))
    s.set(base_pose[i])
    s.pack(fill="x")
    sliders.append((s, i))


def reset():
    global arm_target, prev_arm_q
    for i in range(len(base_pose)):
        base_pose[i] = base_defaults[i]
    arm_target = arm_default.copy()

    # MODIFICATION: Resetting theta3 to -1.0 here as well to maintain "Elbow Up"
    prev_arm_q = np.array([0.0, 0.0, -1.0])

    for s, i in sliders:
        s.set(base_pose[i])

    draw();
    canvas.draw()


tk.Button(panel, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()