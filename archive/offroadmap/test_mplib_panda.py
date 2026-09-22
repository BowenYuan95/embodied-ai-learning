import mplib


urdf = "/home/bowenyuan/miniforge3/envs/embodied/lib/python3.12/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"

srdf = "/home/bowenyuan/miniforge3/envs/embodied/lib/python3.12/site-packages/mani_skill/assets/robots/panda/panda_v2.srdf"


print("Creating mplib Planner...")


planner = mplib.Planner(
    urdf=urdf,
    srdf=srdf,

    user_link_names=[
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
        "panda_hand_tcp",
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_leftfinger_pad",
        "panda_rightfinger_pad",
    ],

    user_joint_names=[
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ],

    move_group="panda_hand_tcp",
)


print("Planner created!")