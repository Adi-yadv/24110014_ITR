import time
import mujoco
import mujoco.viewer
import glfw
import numpy as np


MODEL_PATH = (
    "/home/aditya/robotis_mujoco_menagerie/"
    "robotis_tb3/turtlebot3_waffle_pi.xml"
)


# Load model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# Get wheel actuator IDs
left_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)


# Get ONLY the base body ID
base_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)


SPEED = 5.0


print("\n--- TurtleBot Keyboard Control ---")
print("↑ = Forward")
print("↓ = Backward")
print("← = Turn Left")
print("→ = Turn Right")
print("SPACE = Stop")
print("ESC = Quit\n")


# Keyboard control
def key_callback(keycode):

    if keycode == glfw.KEY_UP:
        data.ctrl[left_motor] = SPEED
        data.ctrl[right_motor] = SPEED

    elif keycode == glfw.KEY_DOWN:
        data.ctrl[left_motor] = -SPEED
        data.ctrl[right_motor] = -SPEED

    elif keycode == glfw.KEY_LEFT:
        data.ctrl[left_motor] = -SPEED
        data.ctrl[right_motor] = SPEED

    elif keycode == glfw.KEY_RIGHT:
        data.ctrl[left_motor] = SPEED
        data.ctrl[right_motor] = -SPEED

    elif keycode == glfw.KEY_SPACE:
        data.ctrl[left_motor] = 0
        data.ctrl[right_motor] = 0


# Launch MuJoCo
with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    # Show frames of all bodies.
    # The size of these can be controlled through XML visual scale.

    last_print_time = 0

    while viewer.is_running():

        # Step physics
        mujoco.mj_step(model, data)

        # ==========================================
        # BASE BODY ROTATION MATRIX ONLY
        # ==========================================

        rotation_matrix = data.xmat[base_body].reshape(3, 3)

        current_time = time.time()

        # Print 5 times per second
        if current_time - last_print_time > 0.2:

            print("\nBase Body Rotation Matrix:")
            print(np.round(rotation_matrix, 3))

            last_print_time = current_time

        # Update viewer
        viewer.sync()

        time.sleep(model.opt.timestep)
