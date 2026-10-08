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
# FIND REVOLUTE JOINTS
# ============================================================

revolute_joints = []

for jid in range(model.njnt):

    if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
        revolute_joints.append(jid)


if len(revolute_joints) < 6:

    raise RuntimeError(
        f"Expected at least 6 revolute joints, "
        f"but found only {len(revolute_joints)}."
    )


joint_ids = revolute_joints[:6]


# ============================================================
# PRINT JOINT INFORMATION
# ============================================================

print()
print("=" * 80)
print("HEAL ROBOT - 6 DOF MUJOCO vs DH FORWARD KINEMATICS")
print("=" * 80)

print()
print("REVOLUTE JOINTS")
print("-" * 80)

for i, jid in enumerate(joint_ids):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        jid
    )

    qpos_address = model.jnt_qposadr[jid]

    print(
        f"q{i+1} : "
        f"{name}    "
        f"qpos[{qpos_address}]"
    )


# ============================================================
# END EFFECTOR SITE
# ============================================================

site_name = "right_center"

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    site_name
)

if site_id == -1:

    raise RuntimeError(
        "Could not find site: right_center"
    )


# ============================================================
# DH PARAMETERS
# ============================================================
#
# Standard DH convention:
#
#             a       alpha       d       theta_offset
#
# Joint 1     0       +90 deg    0.171       0
# Joint 2     0.1735   0 deg     0           0
#
# Joints 3-6 are currently placeholders.
#
# IMPORTANT:
# Replace these four rows with the actual HEAL
# DH parameters once you have them.
#
# ============================================================

DH_TABLE = np.array([

    [0.0000,  np.pi / 2,  0.1710,  0.0000],

    [0.1735,  0.0000,     0.0000,  0.0000],

    [0.0000,  0.0000,     0.0000,  0.0000],

    [0.0000,  0.0000,     0.0000,  0.0000],

    [0.0000,  0.0000,     0.0000,  0.0000],

    [0.0000,  0.0000,     0.0000,  0.0000],
])


# ============================================================
# DH TRANSFORMATION
# ============================================================

def dh_transform(a, alpha, d, theta):

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    ct = np.cos(theta)
    st = np.sin(theta)

    T = np.array([

        [
            ct,
            -st * ca,
            st * sa,
            a * ct
        ],

        [
            st,
            ct * ca,
            -ct * sa,
            a * st
        ],

        [
            0.0,
            sa,
            ca,
            d
        ],

        [
            0.0,
            0.0,
            0.0,
            1.0
        ]

    ])

    return T


# ============================================================
# 6-DOF DH FORWARD KINEMATICS
# ============================================================

def calculate_fk(q):

    T = np.eye(4)

    transforms = []

    for i in range(6):

        a = DH_TABLE[i, 0]

        alpha = DH_TABLE[i, 1]

        d = DH_TABLE[i, 2]

        theta_offset = DH_TABLE[i, 3]

        theta = q[i] + theta_offset

        Ti = dh_transform(
            a,
            alpha,
            d,
            theta
        )

        transforms.append(Ti)

        T = T @ Ti

    return T, transforms


# ============================================================
# GET ACTUAL MUJOCO END-EFFECTOR TRANSFORMATION
# ============================================================

def get_mujoco_transform():

    position = data.site_xpos[site_id].copy()

    rotation = data.site_xmat[
        site_id
    ].reshape(3, 3).copy()

    T = np.eye(4)

    T[:3, :3] = rotation

    T[:3, 3] = position

    return T


# ============================================================
# ROTATION MATRIX TO ROLL-PITCH-YAW
# ============================================================

def rotation_to_rpy(R):

    pitch = np.arcsin(
        np.clip(
            -R[2, 0],
            -1.0,
            1.0
        )
    )

    roll = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    return roll, pitch, yaw


# ============================================================
# GENERATE 6 JOINT ANGLES
# ============================================================

def generate_joint_angles(t):

    q = np.zeros(6)

    q[0] = 0.50 * np.sin(0.50 * t)

    q[1] = 0.40 * np.sin(0.60 * t)

    q[2] = 0.30 * np.sin(0.70 * t)

    q[3] = 0.25 * np.sin(0.80 * t)

    q[4] = 0.20 * np.sin(0.90 * t)

    q[5] = 0.20 * np.sin(1.00 * t)

    return q


# ============================================================
# SET SIX MUJOCO JOINTS
# ============================================================

def set_joint_angles(q):

    for i, jid in enumerate(joint_ids):

        qpos_address = model.jnt_qposadr[jid]

        angle = q[i]

        if model.jnt_limited[jid]:

            lower = model.jnt_range[jid, 0]

            upper = model.jnt_range[jid, 1]

            angle = np.clip(
                angle,
                lower,
                upper
            )

        data.qpos[qpos_address] = angle


# ============================================================
# MAIN SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print()
    print("=" * 80)
    print("6-DOF HEAL FORWARD KINEMATICS")
    print("=" * 80)

    print()
    print("MuJoCo end-effector : right_center")

    print()
    print("WARNING:")
    print("DH parameters for joints 3-6 are placeholders.")
    print("The comparison is therefore mathematical/demo only")
    print("until the real HEAL DH table is inserted.")

    print()
    print("Close the MuJoCo window to stop.")
    print()


    start_time = time.time()

    last_print = 0.0


    while viewer.is_running():

        # ====================================================
        # TIME
        # ====================================================

        t = time.time() - start_time


        # ====================================================
        # GENERATE SIX JOINT ANGLES
        # ====================================================

        q_command = generate_joint_angles(t)


        # ====================================================
        # SET MUJOCO JOINTS
        # ====================================================

        set_joint_angles(q_command)


        # ====================================================
        # MUJOCO FORWARD KINEMATICS
        # ====================================================

        mujoco.mj_forward(
            model,
            data
        )


        # ====================================================
        # READ ACTUAL JOINT ANGLES
        # ====================================================

        q_actual = np.zeros(6)

        for i, jid in enumerate(joint_ids):

            qpos_address = model.jnt_qposadr[jid]

            q_actual[i] = data.qpos[qpos_address]


        # ====================================================
        # ACTUAL MUJOCO END-EFFECTOR TRANSFORM
        # ====================================================

        T_mujoco = get_mujoco_transform()

        R_mujoco = T_mujoco[:3, :3]

        p_mujoco = T_mujoco[:3, 3]


        # ====================================================
        # MUJOCO ROLL PITCH YAW
        # ====================================================

        roll, pitch, yaw = rotation_to_rpy(
            R_mujoco
        )


        # ====================================================
        # DH FORWARD KINEMATICS
        # ====================================================

        T_DH, individual_transforms = calculate_fk(
            q_actual
        )

        R_DH = T_DH[:3, :3]

        p_DH = T_DH[:3, 3]


        # ====================================================
        # POSITION ERROR
        # ====================================================

        position_error_vector = (
            p_mujoco - p_DH
        )

        position_error = np.linalg.norm(
            position_error_vector
        )


        # ====================================================
        # ORIENTATION ERROR
        # ====================================================

        R_error = (
            R_DH.T @ R_mujoco
        )

        angle_argument = (
            np.trace(R_error) - 1.0
        ) / 2.0

        angle_argument = np.clip(
            angle_argument,
            -1.0,
            1.0
        )

        orientation_error = np.arccos(
            angle_argument
        )


        # ====================================================
        # COMPLETE TRANSFORMATION ERROR
        # ====================================================

        T_error = (
            np.linalg.inv(T_DH)
            @ T_mujoco
        )


        # ====================================================
        # PRINT
        # ====================================================

        if t - last_print > 0.5:

            last_print = t

            print(
                "\033[2J\033[H",
                end=""
            )

            print("=" * 80)

            print(
                "HEAL ROBOT - 6 DOF MUJOCO vs DH FK"
            )

            print("=" * 80)


            # ------------------------------------------------
            # JOINT ANGLES
            # ------------------------------------------------

            print()
            print("JOINT ANGLES")
            print("-" * 80)

            for i in range(6):

                print(
                    f"q{i+1} = "
                    f"{q_actual[i]: .5f} rad "
                    f"("
                    f"{np.degrees(q_actual[i]): .2f}"
                    f" deg)"
                )


            # ------------------------------------------------
            # MUJOCO POSITION
            # ------------------------------------------------

            print()
            print("MUJOCO END-EFFECTOR POSITION")
            print("-" * 80)

            print(
                f"X = {p_mujoco[0]: .5f} m"
            )

            print(
                f"Y = {p_mujoco[1]: .5f} m"
            )

            print(
                f"Z = {p_mujoco[2]: .5f} m"
            )


            # ------------------------------------------------
            # DH POSITION
            # ------------------------------------------------

            print()
            print("DH END-EFFECTOR POSITION")
            print("-" * 80)

            print(
                f"X = {p_DH[0]: .5f} m"
            )

            print(
                f"Y = {p_DH[1]: .5f} m"
            )

            print(
                f"Z = {p_DH[2]: .5f} m"
            )


            # ------------------------------------------------
            # POSITION ERROR
            # ------------------------------------------------

            print()
            print("POSITION ERROR")
            print("-" * 80)

            print(
                f"dX = "
                f"{position_error_vector[0]: .5f} m"
            )

            print(
                f"dY = "
                f"{position_error_vector[1]: .5f} m"
            )

            print(
                f"dZ = "
                f"{position_error_vector[2]: .5f} m"
            )

            print(
                f"|e_position| = "
                f"{position_error:.5f} m "
                f"("
                f"{position_error * 1000:.2f}"
                f" mm)"
            )


            # ------------------------------------------------
            # ORIENTATION
            # ------------------------------------------------

            print()
            print("MUJOCO END-EFFECTOR ORIENTATION")
            print("-" * 80)

            print(
                f"Roll  = "
                f"{np.degrees(roll): .3f} deg"
            )

            print(
                f"Pitch = "
                f"{np.degrees(pitch): .3f} deg"
            )

            print(
                f"Yaw   = "
                f"{np.degrees(yaw): .3f} deg"
            )


            # ------------------------------------------------
            # MUJOCO ROTATION MATRIX
            # ------------------------------------------------

            print()
            print("MUJOCO ROTATION MATRIX")
            print("-" * 80)

            print(
                np.array2string(
                    R_mujoco,
                    precision=5,
                    suppress_small=True
                )
            )


            # ------------------------------------------------
            # DH ROTATION MATRIX
            # ------------------------------------------------

            print()
            print("DH ROTATION MATRIX")
            print("-" * 80)

            print(
                np.array2string(
                    R_DH,
                    precision=5,
                    suppress_small=True
                )
            )


            # ------------------------------------------------
            # ORIENTATION ERROR
            # ------------------------------------------------

            print()
            print("ORIENTATION ERROR")
            print("-" * 80)

            print(
                f"|e_rotation| = "
                f"{np.degrees(orientation_error):.4f}"
                f" deg"
            )


            # ------------------------------------------------
            # MUJOCO HOMOGENEOUS TRANSFORMATION
            # ------------------------------------------------

            print()
            print("MUJOCO T")
            print("-" * 80)

            print(
                np.array2string(
                    T_mujoco,
                    precision=4,
                    suppress_small=True
                )
            )


            # ------------------------------------------------
            # DH HOMOGENEOUS TRANSFORMATION
            # ------------------------------------------------

            print()
            print("DH T")
            print("-" * 80)

            print(
                np.array2string(
                    T_DH,
                    precision=4,
                    suppress_small=True
                )
            )


            # ------------------------------------------------
            # TRANSFORMATION ERROR
            # ------------------------------------------------

            print()
            print("T_ERROR = inv(T_DH) @ T_MUJOCO")
            print("-" * 80)

            print(
                np.array2string(
                    T_error,
                    precision=4,
                    suppress_small=True
                )
            )


            print()
            print("=" * 80)
            print(
                "NOTE: DH joints 3-6 are placeholders."
            )
            print("=" * 80)


        # ====================================================
        # UPDATE VIEWER
        # ====================================================

        viewer.sync()

        time.sleep(0.01)
