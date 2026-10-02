import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# FRANKA MODEL
# ============================================================

MODEL_PATH = "../robot_descriptions/franka/panda.xml"


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# END-EFFECTOR BODY
# ============================================================

BODY_NAME = "hand"

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    BODY_NAME
)

if body_id == -1:
    raise RuntimeError(
        "Could not find Franka hand body"
    )


# ============================================================
# OPEN MUJOCO VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print("\n")
    print("=" * 70)
    print("FRANKA - MUJOCO END-EFFECTOR TEST")
    print("=" * 70)

    print("\nClose the MuJoCo window to stop.")

    start_time = time.time()

    last_print = 0

    while viewer.is_running():

        t = time.time() - start_time


        # ====================================================
        # FRANKA JOINT MOTION
        # ====================================================

        data.qpos[0] = 0.5 * np.sin(0.6 * t)

        data.qpos[1] = 0.4 * np.sin(0.7 * t)

        data.qpos[2] = 0.3 * np.sin(0.5 * t)

        data.qpos[3] = -0.5 + 0.3 * np.sin(0.4 * t)

        data.qpos[4] = 0.3 * np.sin(0.8 * t)

        data.qpos[5] = 0.2 * np.sin(0.6 * t)

        data.qpos[6] = 0.2 * np.sin(0.9 * t)


        # ====================================================
        # GRIPPER
        # ====================================================

        data.qpos[7] = 0.02

        data.qpos[8] = 0.02


        # ====================================================
        # MUJOCO FORWARD KINEMATICS
        # ====================================================

        mujoco.mj_forward(
            model,
            data
        )


        # ====================================================
        # HAND POSITION
        # ====================================================

        position = data.xpos[body_id].copy()

        rotation = data.xmat[
            body_id
        ].reshape(3, 3).copy()


        # ====================================================
        # PRINT
        # ====================================================

        if t - last_print > 0.5:

            last_print = t

            print("\033[2J\033[H", end="")

            print("=" * 70)
            print("FRANKA - MUJOCO END-EFFECTOR")
            print("=" * 70)

            print(
                f"\nTime = {t:.2f} s"
            )

            print("\nArm joint angles")
            print("-" * 70)

            for i in range(7):

                print(
                    f"q{i + 1} = "
                    f"{data.qpos[i]: .4f} rad "
                    f"({np.degrees(data.qpos[i]): .2f} deg)"
                )


            print("\nHand position")
            print("-" * 70)

            print(
                f"X = {position[0]: .6f} m"
            )

            print(
                f"Y = {position[1]: .6f} m"
            )

            print(
                f"Z = {position[2]: .6f} m"
            )


            print("\nHand rotation matrix")
            print("-" * 70)

            print(rotation)


        viewer.sync()

        time.sleep(0.01)
