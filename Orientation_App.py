import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def get_rotation_matrix(roll, pitch, yaw):
    """Calculates the rotation matrix using Z-Y-X (Yaw-Pitch-Roll) sequence."""
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Extract the true angle between corresponding axes from the matrix diagonal
    # Clip values between -1 and 1 to prevent floating point nan errors
    diag = np.clip([R[0,0], R[1,1], R[2,2]], -1.0, 1.0)
    true_angles = np.degrees(np.arccos(diag))
    
    return R, true_angles

# --- Setup Figure and Layout ---
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Rigid Body Rotation Visualizer')
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, bottom=0.3, right=0.65, top=0.95)

ax.set_xlim([-1.2, 1.2]); ax.set_ylim([-1.2, 1.2]); ax.set_zlim([-1.2, 1.2])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Yaw, Pitch, Roll Rotation', fontsize=14, weight='bold')

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# Fixed Axes
ax.quiver(0, 0, 0, 1, 0, 0, color='k', alpha=0.3, linewidth=2, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 1, 0, color='k', alpha=0.3, linewidth=2, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, 1, color='k', alpha=0.3, linewidth=2, arrow_length_ratio=0.1)

ax.plot([], [], [], color='k', alpha=0.3, linewidth=2, label='Fixed Frame')
ax.plot([], [], [], color='r', linewidth=3, label='Body X')
ax.plot([], [], [], color='g', linewidth=3, label='Body Y')
ax.plot([], [], [], color='b', linewidth=3, label='Body Z')
ax.legend(loc='upper left', fontsize='medium', framealpha=1.0, edgecolor='black')

body_x_arrow = None
body_y_arrow = None
body_z_arrow = None

# --- UI Elements ---
matrix_text = fig.text(0.68, 0.65, '', fontsize=13, family='monospace', 
                       bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=0.5'))
angle_text = fig.text(0.68, 0.45, '', fontsize=12, color='#004488', 
                      bbox=dict(facecolor='#e6f2ff', edgecolor='#004488', boxstyle='round,pad=0.5'))

ax_roll = plt.axes([0.15, 0.16, 0.45, 0.03], facecolor='#e0e0e0')
ax_pitch = plt.axes([0.15, 0.10, 0.45, 0.03], facecolor='#e0e0e0')
ax_yaw = plt.axes([0.15, 0.04, 0.45, 0.03], facecolor='#e0e0e0')

s_roll = Slider(ax_roll, 'Roll (X) °', -180, 180, valinit=0, valstep=1)
s_pitch = Slider(ax_pitch, 'Pitch (Y) °', -180, 180, valinit=0, valstep=1)
s_yaw = Slider(ax_yaw, 'Yaw (Z) °', -180, 180, valinit=0, valstep=1)

def update(val=None):
    global body_x_arrow, body_y_arrow, body_z_arrow
    
    R, true_angles = get_rotation_matrix(s_roll.val, s_pitch.val, s_yaw.val)
    
    i, j, k = R @ [1,0,0], R @ [0,1,0], R @ [0,0,1]
    
    if body_x_arrow: body_x_arrow.remove()
    if body_y_arrow: body_y_arrow.remove()
    if body_z_arrow: body_z_arrow.remove()
        
    body_x_arrow = ax.quiver(0, 0, 0, i[0], i[1], i[2], color='r', linewidth=4, arrow_length_ratio=0.15)
    body_y_arrow = ax.quiver(0, 0, 0, j[0], j[1], j[2], color='g', linewidth=4, arrow_length_ratio=0.15)
    body_z_arrow = ax.quiver(0, 0, 0, k[0], k[1], k[2], color='b', linewidth=4, arrow_length_ratio=0.15)
    
    R_disp = np.where(np.abs(R) < 1e-5, 0.0, R)
    txt = "Rotation Matrix (R):\n\n"
    for row in R_disp:
        txt += f"[{row[0]:>6.2f} {row[1]:>6.2f} {row[2]:>6.2f}]\n"
    matrix_text.set_text(txt)
    
    angle_text.set_text(f"True Angles Between Axes:\n\n"
                        f"Body X vs Fixed X: {true_angles[0]:>5.1f}°\n"
                        f"Body Y vs Fixed Y: {true_angles[1]:>5.1f}°\n"
                        f"Body Z vs Fixed Z: {true_angles[2]:>5.1f}°")
    
    fig.canvas.draw_idle()

s_roll.on_changed(update)
s_pitch.on_changed(update)
s_yaw.on_changed(update)

update()
plt.show()