import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# HEAL MODEL
# ============================================================

MODEL_PATH = (
    "../robot_descriptions/"
    "single_arm_heal_effort_actuation_rs_mj.xml"
)


# ============================================================
# LOAD MUJOCO
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# EXPLORATORY DH PARAMETERS
# ============================================================
#
# IMPORTANT:
# These are NOT claimed to be the official HEAL DH table.
#
# They are a simplified 2-DOF approximation so that we can
# observe the relationship between joint motion and FK.
#
# q1 = base yaw
# q2 = shoulder pitch
#
# The remaining joints are treated as fixed.
# ============================================================

BASE_HEIGHT = 0.171

L1 = 0.1735
L2 = 0.3000


def dh_transform(a, alpha, d, theta):

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,       d],
        [0,   0,        0,        1]
    ])


# ============================================================
# 2-DOF DH FORWARD KINEMATICS
# ============================================================

def calculate_fk(q1, q2):

    # --------------------------------------------------------
    # Base / Joint 1
    # --------------------------------------------------------

    T01 = dh_transform(
        0.0,
        np.pi / 2,
        BASE_HEIGHT,
        q1
    )

    # --------------------------------------------------------
    # Joint 2
    # --------------------------------------------------------

    T12 = dh_transform(
        L1,
        0.0,
        0.0,
        q2
    )

    # --------------------------------------------------------
    # Fixed arm extension
    # --------------------------------------------------------

    T2E = dh_transform(
        L2,
        0.0,
        0.0,
        0.0
    )

    # --------------------------------------------------------
    # Overall FK
    # --------------------------------------------------------

    T02 = T01 @ T12

    T0E = T02 @ T2E

    return T0E


# ============================================================
# MUJOCO END EFFECTOR
# ============================================================

site_name = "right_center"

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    site_name
)

if site_id == -1:

    raise RuntimeError(
        "Could not find right_center site"
    )


# ============================================================
# OPEN VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print("\n")
    print("=" * 75)
    print("HEAL ROBOT - MUJOCO vs DH FORWARD KINEMATICS")
    print("=" * 75)

    print("\nClose the MuJoCo window to stop.")
    print("\n")

    start_time = time.time()

    last_print = 0

    while viewer.is_running():

        # ====================================================
        # TIME
        # ====================================================

        t = time.time() - start_time


        # ====================================================
        # CREATE SMOOTH JOINT MOTION
        # ====================================================

        # Base rotation
        q1 = 0.8 * np.sin(0.8 * t)

        # Shoulder motion
        q2 = 0.6 * np.sin(0.6 * t)


        # ====================================================
        # SET HEAL JOINTS
        # ====================================================

        data.qpos[0] = q1
        data.qpos[1] = q2

        # Keep remaining joints fixed
        data.qpos[2] = 0.0
        data.qpos[3] = 0.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0


        # ====================================================
        # MUJOCO FORWARD KINEMATICS
        # ====================================================

        mujoco.mj_forward(
            model,
            data
        )


        # ====================================================
        # GET MUJOCO GRIPPER POSITION
        # ====================================================

        mujoco_position = (
            data.site_xpos[site_id].copy()
        )


        # ====================================================
        # OUR DH FORWARD KINEMATICS
        # ====================================================

        T_fk = calculate_fk(
            q1,
            q2
        )

        fk_position = T_fk[:3, 3]


        # ====================================================
        # CALCULATE POSITION ERROR
        # ====================================================

        error_vector = (
            mujoco_position -
            fk_position
        )

        error = np.linalg.norm(
            error_vector
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        if t - last_print > 0.5:

            last_print = t

            print("\033[2J\033[H", end="")

            print("=" * 75)
            print("HEAL ROBOT - MUJOCO vs DH FK")
            print("=" * 75)

            print(
                f"\nTime = {t:.2f} s"
            )

            print("\nJoint angles")
            print("-" * 75)

            print(
                f"q1 = {q1: .4f} rad "
                f"({np.degrees(q1): .2f} deg)"
            )

            print(
                f"q2 = {q2: .4f} rad "
                f"({np.degrees(q2): .2f} deg)"
            )

            print("\nMuJoCo gripper position")
            print("-" * 75)

            print(
                f"X = {mujoco_position[0]: .4f} m"
            )

            print(
                f"Y = {mujoco_position[1]: .4f} m"
            )

            print(
                f"Z = {mujoco_position[2]: .4f} m"
            )

            print("\nDH Forward Kinematics")
            print("-" * 75)

            print(
                f"X = {fk_position[0]: .4f} m"
            )

            print(
                f"Y = {fk_position[1]: .4f} m"
            )

            print(
                f"Z = {fk_position[2]: .4f} m"
            )

            print("\nDifference")
            print("-" * 75)

            print(
                f"dX = {error_vector[0]: .4f} m"
            )

            print(
                f"dY = {error_vector[1]: .4f} m"
            )

            print(
                f"dZ = {error_vector[2]: .4f} m"
            )

            print(
                f"\nPosition error = "
                f"{error:.4f} m "
                f"({error * 1000:.1f} mm)"
            )

            print("\nMuJoCo T:")
            print(
                np.array2string(
                    np.eye(4),
                    precision=3
                )
            )

            print("\nDH FK T:")
            print(
                np.array2string(
                    T_fk,
                    precision=3
                )
            )

            print("\nClose the MuJoCo window to stop.")


        # ====================================================
        # UPDATE VIEWER
        # ====================================================

        viewer.sync()

        time.sleep(0.01)
