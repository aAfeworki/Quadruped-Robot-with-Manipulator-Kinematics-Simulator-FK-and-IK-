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

arm_target = [0.3, 0.0, 0.2]
elbow_sign = 1

leg_limits = {"x":(-0.3,0.3),"y":(-0.3,0.3),"z":(-0.5,-0.05)}
arm_limits = {"x":(-0.3,0.5),"y":(-0.3,0.3),"z":(0,0.5)}

legs = {
    "FR":[0.0,0.0,-0.3],
    "FL":[0.0, 0.0,-0.3],
    "RR":[0.0,0.0,-0.3],
    "RL":[0.0, 0.0,-0.3]
}

defaults_legs = {k:v[:] for k,v in legs.items()}
default_arm = arm_target[:]

base_pos = {
    "FR":np.array([0.3,-0.2,0]),
    "FL":np.array([0.3, 0.2,0]),
    "RR":np.array([-0.3,-0.2,0]),
    "RL":np.array([-0.3, 0.2,0])
}

arm_base = np.array([0.24,0,0])

body = np.array([
    [0.3,-0.2,0],[0.3,0.2,0],
    [-0.3,0.2,0],[-0.3,-0.2,0],[0.3,-0.2,0]
])

# -----------------------------
# IK
# -----------------------------
def ik_leg(x,y,z):
    t1 = np.arctan2(y,-z)
    R = max(np.sqrt(y**2+z**2),1e-6)

    D = x**2 + R**2
    c = np.clip((D-L1**2-L2**2)/(2*L1*L2),-1,1)
    t3 = np.arccos(c)

    t2 = np.arctan2(x,R) - np.arctan2(L2*np.sin(t3),L1+L2*np.cos(t3))
    return t1,t2,t3

def ik_arm(X,Y,Z):
    global elbow_sign
    t1 = np.arctan2(Y,X)

    r = np.sqrt(X**2+Y**2)
    z = Z-Lm1
    D = np.sqrt(r**2+z**2)

    c = np.clip((D**2-Lm2**2-Lm3**2)/(2*Lm2*Lm3),-1,1)
    t3 = elbow_sign*np.arccos(c)

    t2 = np.arctan2(z,r) - np.arctan2(Lm3*np.sin(t3),Lm2+Lm3*np.cos(t3))
    return t1,t2,t3

def fk_leg(t1,t2,t3):
    X = L1*np.sin(t2)+L2*np.sin(t2+t3)
    R = L1*np.cos(t2)+L2*np.cos(t2+t3)
    return np.array([X,R*np.sin(t1),-R*np.cos(t1)])

def fk_arm(t1,t2,t3):
    X=(Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.cos(t1)
    Y=(Lm2*np.cos(t2)+Lm3*np.cos(t2+t3))*np.sin(t1)
    Z=Lm2*np.sin(t2)+Lm3*np.sin(t2+t3)+Lm1
    return np.array([X,Y,Z])

# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')

def draw():
    ax.clear()
    ax.plot(body[:,0],body[:,1],body[:,2],linewidth=3)

    angles={}

    for leg,(x,y,z) in legs.items():
        base=base_pos[leg]
        t1,t2,t3=ik_leg(x,y,z)
        angles[leg]=np.degrees([t1,t2,t3])

        knee = base + [L1*np.sin(t2),
                       L1*np.cos(t2)*np.sin(t1),
                      -L1*np.cos(t2)*np.cos(t1)]
        foot = base + fk_leg(t1,t2,t3)

        ax.plot(*zip(base,knee),marker='o')
        ax.plot(*zip(knee,foot),marker='o')

    t1,t2,t3=ik_arm(*arm_target)
    angles["ARM"]=np.degrees([t1,t2,t3])

    j1=arm_base+[0,0,Lm1]
    j2=j1+[Lm2*np.cos(t2)*np.cos(t1),
           Lm2*np.cos(t2)*np.sin(t1),
           Lm2*np.sin(t2)]
    end=arm_base+fk_arm(t1,t2,t3)

    ax.plot(*zip(arm_base,j1),marker='o')
    ax.plot(*zip(j1,j2),marker='o')
    ax.plot(*zip(j2,end),marker='o')

    txt="         θ1     θ2     θ3\n"
    for k in ["FR","FL","RR","RL","ARM"]:
        a=angles[k]
        txt+=f"{k}: {a[0]:6.1f} {a[1]:6.1f} {a[2]:6.1f}\n"

    ax.text2D(0.02,0.98,txt,transform=ax.transAxes,
              bbox=dict(facecolor='white',alpha=0.8),
              va='top')

    ax.set(xlim=[-0.6,0.6],ylim=[-0.6,0.6],zlim=[-0.6,0.6])

# -----------------------------
# UI
# -----------------------------
root=tk.Tk()
root.title("Compact Quadruped IK + Arm")
root.geometry("1200x650")

canvas=FigureCanvasTkAgg(fig,master=root)
canvas.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

panel=tk.Frame(root)
panel.pack(side=tk.RIGHT,fill=tk.Y)

# ---- LEG SLIDERS  ----
def update_leg(l,i,v):
    legs[l][i]=float(v)
    draw(); canvas.draw_idle()

for leg in legs:
    f=tk.LabelFrame(panel,text=leg)
    f.pack(fill="x",pady=3)

    for i,n in enumerate(["X","Y","Z"]):
        s=tk.Scale(f,from_=leg_limits[n.lower()][0],
                   to=leg_limits[n.lower()][1],
                   resolution=0.01,orient=tk.HORIZONTAL,
                   label=n,
                   command=lambda v,l=leg,j=i:update_leg(l,j,v))
        s.set(legs[leg][i])
        s.pack(side="left",expand=True,fill="x")

# ---- ARM SLIDERS ----
def update_arm(i,v):
    arm_target[i]=float(v)
    draw(); canvas.draw_idle()

f=tk.LabelFrame(panel,text="ARM")
f.pack(fill="x",pady=3)

for i,n in enumerate(["X","Y","Z"]):
    s=tk.Scale(f,from_=arm_limits[n.lower()][0],
               to=arm_limits[n.lower()][1],
               resolution=0.01,orient=tk.HORIZONTAL,
               label=n,
               command=lambda v,j=i:update_arm(j,v))
    s.set(arm_target[i])
    s.pack(side="left",expand=True,fill="x")

# ---- BUTTONS ----
btn_frame = tk.Frame(panel)
btn_frame.pack(fill="x", pady=10)

def toggle_elbow():
    global elbow_sign
    elbow_sign *= -1
    draw(); canvas.draw_idle()

def reset():
    global arm_target
    for k in legs:
        legs[k]=defaults_legs[k][:]
    arm_target = default_arm[:]
    draw(); canvas.draw()

tk.Button(btn_frame,text="Elbow Up/Down",command=toggle_elbow)\
    .pack(side="left",expand=True,fill="x")

tk.Button(btn_frame,text="Reset",command=reset)\
    .pack(side="left",expand=True,fill="x")

# -----------------------------
draw()
canvas.draw()
root.mainloop()