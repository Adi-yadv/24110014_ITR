import time
import math
import numpy as np
import mujoco
import glfw


# ============================================================
# CRAZYFLIE STABILIZED FLIGHT CONTROLLER
# ============================================================

MODEL_PATH = (
    "/home/aditya/Documents/ITR_CS/LAB_1/"
    "mujoco_menagerie/bitcraze_crazyflie_2/cf2.xml"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# IDS
# ============================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "cf2"
)

thrust_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "body_thrust"
)

roll_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "x_moment"
)

pitch_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "y_moment"
)

yaw_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "z_moment"
)


# ============================================================
# PRINT MODEL INFORMATION
# ============================================================

print()
print("================================================")
print("       CRAZYFLIE STABILIZED FLIGHT")
print("================================================")
print()
print("DOF             :", model.nv)
print("Mass            :", model.body_mass[body_id], "kg")
print("Gravity         :", model.opt.gravity)
print()
print("Actuators")
print("-------------------------------")
print("Thrust          :", thrust_id)
print("Roll moment     :", roll_id)
print("Pitch moment    :", pitch_id)
print("Yaw moment      :", yaw_id)
print()


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

g = abs(model.opt.gravity[2])

mass = model.body_mass[body_id]

print("Calculated weight =", mass * g, "N")
print()


# ============================================================
# DESIRED FLIGHT CONDITION
# ============================================================

TARGET_ALTITUDE = 0.25       # 25 cm above ground

MAX_ROLL = math.radians(12)
MAX_PITCH = math.radians(12)

YAW_RATE = math.radians(45)


# ============================================================
# ATTITUDE CONTROLLER
# ============================================================

# Proportional attitude gain
K_ROLL = 0.12
K_PITCH = 0.12
K_YAW = 0.08

# Angular-rate damping
K_ROLL_RATE = 0.035
K_PITCH_RATE = 0.035
K_YAW_RATE = 0.025


# ============================================================
# ALTITUDE CONTROLLER
# ============================================================

K_ALTITUDE = 4.0
K_VERTICAL_RATE = 1.5


# ============================================================
# INITIAL TARGET ATTITUDE
# ============================================================

target_roll = 0.0
target_pitch = 0.0
target_yaw = 0.0


# ============================================================
# ANGLE UTILITIES
# ============================================================

def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_to_rpy(R):

    # ZYX Euler angles
    #
    # R = Rz(yaw) Ry(pitch) Rx(roll)

    pitch = math.asin(
        np.clip(-R[2, 0], -1.0, 1.0)
    )

    roll = math.atan2(
        R[2, 1],
        R[2, 2]
    )

    yaw = math.atan2(
        R[1, 0],
        R[0, 0]
    )

    return roll, pitch, yaw


# ============================================================
# RESET
# ============================================================

def reset_simulation():

    global target_roll
    global target_pitch
    global target_yaw

    mujoco.mj_resetData(model, data)

    # Start 25 cm above ground.
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = TARGET_ALTITUDE

    # Identity quaternion
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Zero velocity
    data.qvel[:] = 0.0

    target_roll = 0.0
    target_pitch = 0.0
    target_yaw = 0.0

    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    print()
    print("========== RESET ==========")
    print("Starting altitude:", TARGET_ALTITUDE, "m")


# ============================================================
# GLFW
# ============================================================

if not glfw.init():
    raise RuntimeError("GLFW initialization failed")


window = glfw.create_window(
    1200,
    800,
    "Crazyflie - Stabilized Flight",
    None,
    None
)

if window is None:
    glfw.terminate()
    raise RuntimeError("Could not create GLFW window")


glfw.make_context_current(window)

# No artificial monitor synchronization.
glfw.swap_interval(0)


# ============================================================
# MUJOCO RENDERING
# ============================================================

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

scene = mujoco.MjvScene(
    model,
    maxgeom=10000
)

context = mujoco.MjrContext(
    model,
    mujoco.mjtFontScale.mjFONTSCALE_150
)


# Camera
cam.lookat[:] = [0.0, 0.0, 0.15]
cam.distance = 1.0
cam.azimuth = 135
cam.elevation = -15

# Show sites
opt.sitegroup[:] = 1


# ============================================================
# INITIALIZE
# ============================================================

reset_simulation()


# ============================================================
# KEYBOARD STATE
# ============================================================

previous_p = False

last_time = time.time()
last_print = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while not glfw.window_should_close(window):

        glfw.poll_events()

        # ----------------------------------------------------
        # ESC
        # ----------------------------------------------------

        if glfw.get_key(
            window,
            glfw.KEY_ESCAPE
        ) == glfw.PRESS:

            break


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        p_pressed = (
            glfw.get_key(
                window,
                glfw.KEY_P
            ) == glfw.PRESS
        )

        if p_pressed and not previous_p:
            reset_simulation()

        previous_p = p_pressed


        # ----------------------------------------------------
        # CURRENT ORIENTATION
        # ----------------------------------------------------

        R = data.xmat[body_id].reshape(3, 3)

        roll, pitch, yaw = rotation_to_rpy(R)


        # ----------------------------------------------------
        # KEYBOARD ATTITUDE COMMAND
        # ----------------------------------------------------

        # Roll

        if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:

            target_roll = MAX_ROLL

        elif glfw.get_key(window, glfw.KEY_L) == glfw.PRESS:

            target_roll = -MAX_ROLL

        else:

            # Return to level when released
            target_roll = 0.0


        # Pitch

        if glfw.get_key(window, glfw.KEY_U) == glfw.PRESS:

            target_pitch = MAX_PITCH

        elif glfw.get_key(window, glfw.KEY_O) == glfw.PRESS:

            target_pitch = -MAX_PITCH

        else:

            target_pitch = 0.0


        # ----------------------------------------------------
        # YAW
        # ----------------------------------------------------

        dt = model.opt.timestep

        yaw_left = (
            glfw.get_key(
                window,
                glfw.KEY_N
            ) == glfw.PRESS
        )

        yaw_right = (
            glfw.get_key(
                window,
                glfw.KEY_M
            ) == glfw.PRESS
        )

        if yaw_left:

            target_yaw += YAW_RATE * dt

        if yaw_right:

            target_yaw -= YAW_RATE * dt

        target_yaw = wrap_angle(target_yaw)


        # ----------------------------------------------------
        # ANGULAR VELOCITY
        # ----------------------------------------------------

        # qvel[3:6] is angular velocity in world coordinates.
        omega_world = data.qvel[3:6]

        # Convert angular velocity to body coordinates.
        omega_body = R.T @ omega_world

        wx = omega_body[0]
        wy = omega_body[1]
        wz = omega_body[2]


        # ----------------------------------------------------
        # ATTITUDE ERROR
        # ----------------------------------------------------

        roll_error = wrap_angle(
            target_roll - roll
        )

        pitch_error = wrap_angle(
            target_pitch - pitch
        )

        yaw_error = wrap_angle(
            target_yaw - yaw
        )


        # ----------------------------------------------------
        # PD ATTITUDE CONTROL
        # ----------------------------------------------------

        roll_command = (
            K_ROLL * roll_error
            - K_ROLL_RATE * wx
        )

        pitch_command = (
            K_PITCH * pitch_error
            - K_PITCH_RATE * wy
        )

        yaw_command = (
            K_YAW * yaw_error
            - K_YAW_RATE * wz
        )


        # ----------------------------------------------------
        # LIMIT MOMENTS
        # ----------------------------------------------------

        roll_command = np.clip(
            roll_command,
            -0.25,
            0.25
        )

        pitch_command = np.clip(
            pitch_command,
            -0.25,
            0.25
        )

        yaw_command = np.clip(
            yaw_command,
            -0.20,
            0.20
        )


        # ----------------------------------------------------
        # ALTITUDE CONTROL
        # ----------------------------------------------------

        z = data.xpos[body_id][2]

        vertical_velocity = data.qvel[2]

        altitude_error = (
            TARGET_ALTITUDE - z
        )

        desired_vertical_acceleration = (
            K_ALTITUDE * altitude_error
            - K_VERTICAL_RATE * vertical_velocity
        )

        required_force = mass * (
            g + desired_vertical_acceleration
        )


        # ----------------------------------------------------
        # COMPENSATE FOR TILT
        # ----------------------------------------------------

        # Body Z axis projection onto world Z.
        vertical_component = R[2, 2]

        vertical_component = max(
            vertical_component,
            0.35
        )

        thrust = (
            required_force
            / vertical_component
        )


        # Crazyflie thrust actuator range.
        thrust = np.clip(
            thrust,
            0.0,
            0.35
        )


        # ----------------------------------------------------
        # SEND COMMANDS
        # ----------------------------------------------------

        data.ctrl[thrust_id] = thrust

        data.ctrl[roll_id] = roll_command

        data.ctrl[pitch_id] = pitch_command

        data.ctrl[yaw_id] = yaw_command


        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )


        # ----------------------------------------------------
        # UPDATE SCENE
        # ----------------------------------------------------

        mujoco.mjv_updateScene(
            model,
            data,
            opt,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene
        )


        # ----------------------------------------------------
        # RENDER
        # ----------------------------------------------------

        width, height = glfw.get_framebuffer_size(
            window
        )

        viewport = mujoco.MjrRect(
            0,
            0,
            width,
            height
        )

        mujoco.mjr_render(
            viewport,
            scene,
            context
        )

        glfw.swap_buffers(window)


        # ----------------------------------------------------
        # TERMINAL DISPLAY
        # ----------------------------------------------------

        now = time.time()

        if now - last_print > 0.15:

            R = data.xmat[body_id].reshape(
                3,
                3
            )

            position = data.xpos[body_id]

            print("\033[H\033[J", end="")

            print("================================================")
            print("             CRAZYFLIE FLIGHT")
            print("================================================")

            print()
            print(
                "POSITION [m]"
            )

            print(
                f"X = {position[0]: .4f}"
                f"    Y = {position[1]: .4f}"
                f"    Z = {position[2]: .4f}"
            )

            print()
            print("ATTITUDE")
            print("--------------------------------")

            print(
                f"ROLL  = {math.degrees(roll): .3f} deg"
            )

            print(
                f"PITCH = {math.degrees(pitch): .3f} deg"
            )

            print(
                f"YAW   = {math.degrees(yaw): .3f} deg"
            )

            print()
            print("TARGET")
            print("--------------------------------")

            print(
                f"ROLL  = {math.degrees(target_roll): .3f} deg"
            )

            print(
                f"PITCH = {math.degrees(target_pitch): .3f} deg"
            )

            print(
                f"YAW   = {math.degrees(target_yaw): .3f} deg"
            )

            print()
            print("ROTATION MATRIX R")
            print("--------------------------------")

            for row in R:

                print(
                    "[ "
                    + "  ".join(
                        f"{value: .5f}"
                        for value in row
                    )
                    + " ]"
                )

            print("--------------------------------")

            print()
            print("CONTROL")
            print("--------------------------------")

            print(
                f"Thrust = {thrust:.5f}"
            )

            print(
                f"Roll moment  = {roll_command:+.5f}"
            )

            print(
                f"Pitch moment = {pitch_command:+.5f}"
            )

            print(
                f"Yaw moment   = {yaw_command:+.5f}"
            )

            print()
            print("KEYBOARD")
            print("--------------------------------")
            print("J / L : Roll")
            print("U / O : Pitch")
            print("N / M : Yaw")
            print("P     : Reset")
            print("ESC   : Quit")
            print()
            print("Release J/L/U/O -> return to level")

            last_print = now


        # ----------------------------------------------------
        # SMALL REAL-TIME DELAY
        # ----------------------------------------------------

        time.sleep(0.001)


finally:

    context.free()

    glfw.destroy_window(
        window
    )

    glfw.terminate()

    print()
    print("Crazyflie simulation closed.")


