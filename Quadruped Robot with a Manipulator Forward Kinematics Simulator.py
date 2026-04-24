import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Parameters
# -----------------------------
L1, L2 = 0.2, 0.2
Lm1, Lm2, Lm3 = 0.02, 0.2, 0.2

# -----------------------------
# Unified Robot Definition
# -----------------------------
robot = {
    "FR": {"q":[0.0,-0.5,1.0], "base":[0.3,-0.2,0], "type":"leg"},
    "FL": {"q":[0.0,-0.5,1.0], "base":[0.3, 0.2,0], "type":"leg"},
    "RR": {"q":[0.0,-0.5,1.0], "base":[-0.3,-0.2,0], "type":"leg"},
    "RL": {"q":[0.0,-0.5,1.0], "base":[-0.3, 0.2,0], "type":"leg"},
    "ARM":{"q":[0,np.pi/3,-np.pi/3], "base":[0.24,0,0], "type":"arm"}
}

defaults = {k: v["q"][:] for k,v in robot.items()}

limits = {
    "leg":[(-np.pi/6,np.pi/6),(-np.pi/2,np.pi/6),(0,np.pi)],
    "arm":[(-np.pi,np.pi),(0,np.pi),(-np.pi,np.pi)]
}

body = np.array([
    [0.3,-0.2,0],[0.3,0.2,0],
    [-0.3,0.2,0],[-0.3,-0.2,0],[0.3,-0.2,0]
])

# -----------------------------
# FK
# -----------------------------
def fk_leg(t1,t2,t3):
    X = L1*np.sin(t2) + L2*np.sin(t2+t3)
    R = L1*np.cos(t2) + L2*np.cos(t2+t3)
    return np.array([X, R*np.sin(t1), -R*np.cos(t1)])

def fk_arm(t1,t2,t3):
    X = (Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.cos(t1)
    Y = (Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.sin(t1)
    Z = Lm2*np.sin(t2)+Lm3*np.sin(t2+t3)+Lm1
    return np.array([X,Y,Z])

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

def draw():
    ax.clear()
    ax.plot(body[:,0], body[:,1], body[:,2], linewidth=3)

    for name, r in robot.items():
        t1,t2,t3 = r["q"]
        base = np.array(r["base"])

        if r["type"] == "leg":
            knee = base + np.array([
                L1*np.sin(t2),
                L1*np.cos(t2)*np.sin(t1),
                -L1*np.cos(t2)*np.cos(t1)
            ])
            foot = base + fk_leg(t1,t2,t3)

            ax.plot(*zip(base,knee), marker='o')
            ax.plot(*zip(knee,foot), marker='o')

        else:  # arm
            j1 = base + [0,0,Lm1]
            j2 = j1 + [
                Lm2*np.cos(t2)*np.cos(t1),
                Lm2*np.cos(t2)*np.sin(t1),
                Lm2*np.sin(t2)
            ]
            end = base + fk_arm(t1,t2,t3)

            ax.plot(*zip(base,j1), marker='o')
            ax.plot(*zip(j1,j2), marker='o')
            ax.plot(*zip(j2,end), marker='o')

    ax.set(xlim=[-0.6,0.6], ylim=[-0.6,0.6], zlim=[-0.6,0.6])

# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Quadruped Manipulator")
root.geometry("1100x650")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel = tk.Frame(root)
panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

sliders = []

def update(name,i,val):
    robot[name]["q"][i] = float(val)
    draw(); canvas.draw_idle()

# Create sliders 
for name, r in robot.items():
    f = tk.LabelFrame(panel, text=name)
    f.pack(fill="x", pady=5)

    lim = limits[r["type"]]

    for i in range(3):
        s = tk.Scale(
            f,
            from_=lim[i][0], to=lim[i][1],
            resolution=0.01, orient=tk.HORIZONTAL,
            label=f"θ{i+1}",
            command=lambda v,n=name,j=i: update(n,j,v)
        )
        s.set(r["q"][i])
        s.pack(side="left", expand=True, fill="x")

        sliders.append((s,name,i))

# -----------------------------
# Reset
# -----------------------------
def reset():
    for name in robot:
        robot[name]["q"] = defaults[name][:]
    for s,n,i in sliders:
        s.set(robot[n]["q"][i])

    draw(); canvas.draw()

tk.Button(panel, text="Reset", command=reset).pack(fill="x", pady=10)

# -----------------------------
draw()
canvas.draw()
root.mainloop()