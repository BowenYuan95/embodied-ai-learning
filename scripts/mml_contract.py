"""Data contract for Lesson 3.8 (multimodal policy transition).

This module is the **single source of truth** for the 3.8 data contract. Notebooks
``3.8a``-``3.8d`` import it instead of re-declaring the loader, the tokenizer, or the
counterfactual instruction modes, so the field layout and the language vocabulary
cannot drift between notebooks.

Every field slice is read from the HDF5 file's own ``state_fields`` attribute, never
from hard-coded offsets. The flatten order is task dependent: PickCube puts
``extra.is_grasped`` at ``[18:19]`` so its ``tcp_pose`` starts at 19, while PushCube
has no such field and starts at 18. Reusing one task's offsets on another still yields
legal shapes with **wrong semantics**, so the schema has to travel with the data.

Conventions fixed here, and depended upon by every later section:

``o_t`` is the state **before** ``a_t``
    ``observations`` has ``T + 1`` rows and ``actions`` ``T``, so ``o_t`` / ``a_t`` /
    ``o_{t+1}`` line up index by index. Nothing in the file carries timestamps.
``action_chunk [H, ACTION_DIM]``
    Frozen from lesson 3.7. Lesson 3.8 changes the input side only, so any difference
    measured later can be attributed to modality rather than to action representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_HW = 128
CHANNELS = 3
ACTION_DIM = 8
H = 8                                             # frozen from 3.7
KEEP = {"agent.qpos", "agent.qvel", "extra.tcp_pose", "extra.goal_pos"}

DATASETS = {
    "PickCube-v1": PROJECT_ROOT / "datasets" / "pickcube" / "expert_episodes_rgb.h5",
    "PushCube-v1": PROJECT_ROOT / "datasets" / "pushcube" / "expert_episodes_rgb.h5",
}

# The instruction strings belong to the data contract, not to one subsection: every
# episode-level and task-level use reads them from here.
INSTRUCTIONS = {
    "PickCube-v1": "pick up the cube",
    "PushCube-v1": "push the cube to the goal",
}

OTHER = {"PickCube-v1": "PushCube-v1", "PushCube-v1": "PickCube-v1"}

TASKS = tuple(sorted(DATASETS))


def load_rgb_dataset(env_id):
    """Load one task. Every field slice comes from the FILE, never from hard-coded offsets."""
    with h5py.File(DATASETS[env_id], "r") as f:
        assert "rgb" in f.attrs["obs_mode"], f.attrs["obs_mode"]
        assert "state" in f.attrs["obs_mode"], f.attrs["obs_mode"]
        assert f.attrs["control_mode"] == "pd_joint_pos", f.attrs["control_mode"]
        assert f.attrs["transition_schema_version"] == 2
        assert list(f.attrs["image_shape"]) == [IMAGE_HW, IMAGE_HW, CHANNELS], f.attrs["image_shape"]
        assert f.attrs["image_dtype"] == "uint8", f.attrs["image_dtype"]

        schema = json.loads(f.attrs["state_fields"])
        state_dim = int(f.attrs["state_dim"])

        def spans(field):
            entry = next((x for x in schema if x["field"] == field), None)
            if entry is None:
                raise KeyError(f"{env_id} declares no field {field!r}")
            return entry["start"], entry["stop"]

        proprio_slices = (spans("agent.qpos"), spans("agent.qvel"), spans("extra.tcp_pose"))
        goal_slice = spans("extra.goal_pos")
        excluded = tuple((x["start"], x["stop"]) for x in schema if x["field"] not in KEEP)

        # No dimension may be silently dropped or counted twice.
        covered = (sum(hi - lo for lo, hi in proprio_slices)
                   + goal_slice[1] - goal_slice[0]
                   + sum(hi - lo for lo, hi in excluded))
        assert covered == state_dim, (env_id, covered, state_dim)

        attrs = {k: f.attrs[k] for k in
                 ("obs_mode", "image_camera", "image_shape", "image_dtype", "control_mode",
                  "transition_schema_version", "control_freq_hz", "timestamp_source")}

        episodes = []
        for name in sorted(f.keys()):
            g = f[name]
            obs, act, img = g["observations"][:], g["actions"][:], g["images"][:]
            assert obs.shape[1] == state_dim, (env_id, name, obs.shape)
            assert act.shape[1] == ACTION_DIM, (env_id, name, act.shape)
            assert img.shape == (obs.shape[0], IMAGE_HW, IMAGE_HW, CHANNELS), (env_id, name, img.shape)
            assert obs.shape[0] == act.shape[0] + 1, (env_id, name, obs.shape[0], act.shape[0])
            assert img.dtype == np.uint8, img.dtype
            episodes.append(dict(env_id=env_id, name=name, all_state=obs, state=obs[:-1],
                                 next_state=obs[1:], action=act, image=img,
                                 seed=int(g.attrs["seed"]), success=bool(g.attrs["success"]),
                                 proprio_slices=proprio_slices, goal_slice=goal_slice))

    return dict(env_id=env_id, path=DATASETS[env_id], schema=schema, state_dim=state_dim,
                proprio_slices=proprio_slices, goal_slice=goal_slice, excluded=excluded,
                attrs=attrs, episodes=episodes)


def load_datasets():
    """Load every task in ``DATASETS``."""
    return {env_id: load_rgb_dataset(env_id) for env_id in TASKS}


def paired_episode_lists(datasets=None):
    """``(pick, push, T_common)``: the two episode lists and the common truncation length.

    ``T_common`` is the only index range over which the two tasks can be compared
    frame by frame, so every cross-task measurement in 3.8.4 starts from it.
    """
    if datasets is None:
        datasets = load_datasets()
    pick = datasets["PickCube-v1"]["episodes"]
    push = datasets["PushCube-v1"]["episodes"]
    T_common = min(min(len(e["action"]) for e in pick), min(len(e["action"]) for e in push))
    return pick, push, T_common


# --------------------------------------------------------------------------- #
# Language. A whitespace tokenizer over the union of both instructions; a real
# tokenizer (BPE and friends) is deferred to 3.8.7. What is needed here is a
# language channel that is well shaped and identical in form for every task.
# --------------------------------------------------------------------------- #
VOCAB = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
for _env_id in TASKS:
    for _word in INSTRUCTIONS[_env_id].split():
        VOCAB.setdefault(_word, len(VOCAB))

T_TXT = max(len(INSTRUCTIONS[e].split()) for e in TASKS) + 2      # <bos> + words + <eos>
PAD = VOCAB["<pad>"]


def tokenize(text):
    """Whitespace tokenizer with ``<bos>``/``<eos>``, right-padded to ``T_TXT``."""
    ids = [VOCAB["<bos>"]] + [VOCAB[w] for w in text.split()] + [VOCAB["<eos>"]]
    assert len(ids) <= T_TXT, (text, len(ids), T_TXT)
    return np.asarray(ids + [PAD] * (T_TXT - len(ids)), dtype=np.int64)


LANGUAGE_IDS = {env_id: tokenize(INSTRUCTIONS[env_id]) for env_id in TASKS}


# --------------------------------------------------------------------------- #
# Samples.
# --------------------------------------------------------------------------- #
def proprio_of(ep, t):
    """Extract the proprioception block using the episode's OWN schema."""
    return np.concatenate(
        [ep["all_state"][t][lo:hi] for lo, hi in ep["proprio_slices"]]
    ).astype(np.float32)


def build_sample(ep, t, H=H):
    """One multimodal sample.

    The slices come from the episode, so this works for every task in ``datasets``
    rather than only for the one whose offsets happened to be written down.
    """
    assert 0 <= t <= len(ep["action"]) - H, f"t={t} leaves no room for H={H}"
    lo, hi = ep["goal_slice"]
    return dict(
        image=ep["image"][t],                                    # [128,128,3] uint8
        language_ids=LANGUAGE_IDS[ep["env_id"]],                 # [T_txt]
        proprio=proprio_of(ep, t),                               # [25]
        task_goal=ep["all_state"][t][lo:hi].astype(np.float32),  # [3]
        action_chunk=ep["action"][t:t + H],                      # [8,8]
    )


# --------------------------------------------------------------------------- #
# Counterfactual instruction modes (3.8.4.6).
#
# The three tests have to be three DIFFERENT operations or the ladder collapses:
# with two tasks, "the wrong instruction" IS "the other instruction", so a T2 built
# as a task swap is identical to T3. T2 therefore permutes word ORDER (same bag of
# words) and T3 changes the BAG. A genuine three-rung ladder needs three or more
# instructions -- a data-design constraint, not a code one.
# --------------------------------------------------------------------------- #
def shuffle_words(text, seed=0):
    """Permute the CONTENT words of one instruction: same bag of words, different order."""
    words = text.split()
    perm = np.random.default_rng(seed).permutation(len(words))
    if len(words) > 1 and np.array_equal(perm, np.arange(len(words))):
        perm = np.roll(perm, 1)                       # guarantee a real reordering
    return " ".join(words[i] for i in perm)


def instruction_variants(env_id):
    """T1 correct / T2 shuffled (word order) / T3 contradictory (the bag), plus dropout."""
    correct_text = INSTRUCTIONS[env_id]
    shuffled_text = shuffle_words(correct_text, seed=0)
    assert shuffled_text != correct_text, "shuffle did not reorder anything"
    return {
        "correct": tokenize(correct_text),                    # T1, weakest: fit only
        "shuffled": tokenize(shuffled_text),                   # T2: word ORDER
        "contradictory": LANGUAGE_IDS[OTHER[env_id]].copy(),   # T3, strongest: the BAG
        "drop": np.full(T_TXT, PAD, np.int64),                 # 3.8.5's training dropout
    }
