"""Minimal multimodal policy for Lesson 3.8.5 (fusion model).

The first model that takes ``(image, proprio, goal, language)`` and returns an
``action_chunk [H, ACTION_DIM]``. It is deliberately the simplest thing that can
work, so that a failure is attributable:

    image    -> small CNN -> 256 ┐
    proprio  -> MLP       -> 128 │
    goal     -> MLP       ->  32 ├─ concat -> MLP -> action head -> [H, 8]
    language -> embedding ->  32 ┘

**Concatenation fusion, no attention, on purpose.** This architecture can do
*action-mode* grounding (the instruction selects a behaviour pattern) and
structurally cannot do *referential* grounding (binding a word to a region),
because a mean-pooled language vector never indexes into the visual feature map.
See lesson 3.8.4.5: pooling the image to one vector removes the token grid, so
``K`` and ``V`` would each hold a single token and the softmax would degenerate to
``[1]``. That is a property of the architecture, not a tuning problem. Introducing
cross-attention here would only add a variable to a pipeline that has not been
validated yet.

Two conventions inherited from lesson 3.7 and not to be broken:

``action_chunk [H, 8]`` is frozen
    The output side is byte-for-byte 3.7's contract, so any difference measured in
    3.8.6 is attributable to the input modalities.
the first chunk row is the only fair single-step comparison
    Interiors of a long and a short chunk are averages over different numbers of
    terms, so they are not comparable; ``h = 0`` is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mml_contract as mmc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIELDS = ("image", "proprio", "goal", "language_ids", "task_index", "action_chunk")

# The two language representations of lesson 3.8.4.4:
#   "tokens"  -> L1, a learned embedding per word, masked mean pool
#   "task_id" -> L0, a learned embedding per task
# They are informationally equivalent with two tasks (a bijection exists), which is
# exactly why the L0 arm is a CEILING on the L1 arm and C > B would be a broken
# experiment. Keeping both implementations is what makes that claim testable.
LANGUAGE_MODES = ("tokens", "task_id")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_dataset(datasets, H=mmc.H, t_only=None):
    """Chunk samples for every task, built strictly inside each episode.

    The last ``H - 1`` steps of every episode are dropped rather than padded, so no
    chunk crosses an episode boundary. ``episode_of`` keeps the split episode-level;
    ``task_of`` records the task so the split can hold out one episode per task.

    ``t_only`` restricts the start indices, e.g. ``t_only={0}`` gives exactly one
    sample per episode. The 3.8.6 restricted probe needs that: at ``t = 0`` the two
    tasks' proprioception is *identical*, so a proprio-only model provably cannot
    tell them apart, and the instruction is the only channel that can.
    """
    acc = {k: [] for k in FIELDS}
    episode_of, task_of, start_of = [], [], []
    for env_id in sorted(datasets):
        task_index = sorted(datasets).index(env_id)
        for ei, ep in enumerate(datasets[env_id]["episodes"]):
            T = len(ep["action"])
            assert T - H + 1 > 0, f"{env_id}/{ep['name']}: T={T} is shorter than H={H}"
            for t in range(T - H + 1):
                if t_only is not None and t not in t_only:
                    continue
                s = mmc.build_sample(ep, t, H)
                acc["image"].append(s["image"])
                acc["proprio"].append(s["proprio"])
                acc["goal"].append(s["task_goal"])
                acc["language_ids"].append(s["language_ids"])
                acc["task_index"].append(task_index)
                acc["action_chunk"].append(s["action_chunk"])
                episode_of.append(ei)
                task_of.append(env_id)
                start_of.append(t)

    out = {k: np.stack(v) for k, v in acc.items()}
    out["episode_of"] = np.asarray(episode_of)
    out["task_of"] = np.asarray(task_of)
    out["start_of"] = np.asarray(start_of)

    # The property the loop exists to guarantee: every chunk is exactly its own
    # episode's slice, and never reaches past the end of that episode.
    for i in range(len(out["start_of"])):
        ep = datasets[out["task_of"][i]]["episodes"][out["episode_of"][i]]
        t = int(out["start_of"][i])
        assert t + H <= len(ep["action"]), (i, t, H, len(ep["action"]))
        assert np.array_equal(out["action_chunk"][i], ep["action"][t:t + H]), i
        assert np.array_equal(out["proprio"][i], mmc.proprio_of(ep, t)), i
    return out


def episode_split(datasets, dataset, seed=42, hold_out=1):
    """Episode-level split, holding out ``order[0]`` of **each** task.

    Lesson 3.7 held out one episode of the single task and recorded that
    ``default_rng(42).permutation(5)[0] == 4``. The same rule is applied per task
    here, so the held-out episodes stay the ones earlier lessons identified.
    """
    train_mask = np.zeros(len(dataset["start_of"]), bool)
    val_mask = np.zeros(len(dataset["start_of"]), bool)
    held = {}
    for env_id in sorted(datasets):
        n = len(datasets[env_id]["episodes"])
        order = np.random.default_rng(seed).permutation(n)
        val_ids, train_ids = order[:hold_out], order[hold_out:]
        held[env_id] = sorted(val_ids.tolist())
        is_task = dataset["task_of"] == env_id
        val_mask |= is_task & np.isin(dataset["episode_of"], val_ids)
        train_mask |= is_task & np.isin(dataset["episode_of"], train_ids)

    assert not (train_mask & val_mask).any(), "an episode is in both splits"
    assert (train_mask | val_mask).all(), "a sample was left out of both splits"
    for env_id in sorted(datasets):
        n = len(datasets[env_id]["episodes"])
        assert len(held[env_id]) == hold_out and n - hold_out > 0
    return train_mask, val_mask, held


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def fit_normalization(datasets, held):
    """Fit on the TRAINING episodes' raw frames only -- never on chunk samples,
    never on the validation episodes.

    **Pooled across tasks, and that is load-bearing.** Normalising per task would
    subtract exactly the between-task difference that carries the whole
    task-conditioned signal: measured in 3.8.4.5, the gripper's mean action is
    ``+0.5120`` for PickCube against ``-1.0000`` for PushCube, and 99.54% of the
    task-dependent variance lives in that one dimension. Per-task normalisation
    would map both to zero mean and erase it.
    """
    prop, goal, act = [], [], []
    for env_id in sorted(datasets):
        val_ids = set(held[env_id])
        for ei, ep in enumerate(datasets[env_id]["episodes"]):
            if ei in val_ids:
                continue
            prop.append(np.stack([mmc.proprio_of(ep, t) for t in range(len(ep["action"]))]))
            lo, hi = ep["goal_slice"]
            goal.append(ep["all_state"][:-1, lo:hi].astype(np.float32))
            act.append(ep["action"].astype(np.float32))

    prop = np.concatenate(prop).astype(np.float64)
    goal = np.concatenate(goal).astype(np.float64)
    act = np.concatenate(act).astype(np.float64)

    def stats(x):
        return torch.tensor(x.mean(0), dtype=torch.float32), \
               torch.tensor(np.maximum(x.std(0), 1e-6), dtype=torch.float32)

    p_mean, p_std = stats(prop)
    g_mean, g_std = stats(goal)
    a_mean, a_std = stats(act)
    return dict(proprio=(p_mean, p_std), goal=(g_mean, g_std), action=(a_mean, a_std),
                n_frames=len(prop))


def apply_normalization(dataset, norm):
    """Return a copy with float32 proprio/goal/action scaled; images stay uint8 in [0,255]."""
    out = dict(dataset)
    p_mean, p_std = (t.numpy() for t in norm["proprio"])
    g_mean, g_std = (t.numpy() for t in norm["goal"])
    a_mean, a_std = (t.numpy() for t in norm["action"])
    out["proprio"] = ((dataset["proprio"] - p_mean) / p_std).astype(np.float32)
    out["goal"] = ((dataset["goal"] - g_mean) / g_std).astype(np.float32)
    out["action_chunk"] = ((dataset["action_chunk"] - a_mean) / a_std).astype(np.float32)
    return out


def denormalize_action(y, norm):
    """Map normalised actions back to raw simulator units, for reporting."""
    a_mean, a_std = norm["action"]
    return torch.as_tensor(y) * a_std + a_mean


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _conv_block(cin, cout, stride):
    """Conv + ReLU. No BatchNorm on purpose: one fewer train/eval-mode difference."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.ReLU(inplace=True))


class MultimodalPolicy(nn.Module):
    """Encoder + fusion + action head. Concatenation fusion, no attention."""

    def __init__(self, vocab_size, t_txt, num_tasks=2, language_mode="tokens",
                 action_dim=mmc.ACTION_DIM, horizon=mmc.H,
                 proprio_dim=25, goal_dim=3, embed_dim=16,
                 image_feature=256, proprio_feature=128, goal_feature=32, language_feature=32,
                 fusion_hidden=512, fusion_out=256,
                 use_image=True, use_proprio=True, use_goal=True, use_language=True):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.flags = dict(use_image=use_image, use_proprio=use_proprio,
                          use_goal=use_goal, use_language=use_language)
        assert any(self.flags.values()), "a model with no input modality cannot be trained"
        assert language_mode in LANGUAGE_MODES, language_mode
        self.language_mode = language_mode

        if use_image:
            # Four stride-2 blocks take 128x128 down to 8x8, then global average
            # pooling. Flattening 3*128*128 = 49152 straight into a Linear would
            # cost 12.6M weights in that layer alone and would throw away the
            # spatial inductive bias that makes a CNN the right tool here.
            self.image_encoder = nn.Sequential(
                _conv_block(3, 16, 2),        # 128 -> 64
                _conv_block(16, 32, 2),       #  64 -> 32
                _conv_block(32, 64, 2),       #  32 -> 16
                _conv_block(64, 128, 2),      #  16 ->  8
                nn.AdaptiveAvgPool2d(1),      #   8 ->  1  (global average pooling)
                nn.Flatten(),
                nn.Linear(128, image_feature),
                nn.ReLU(inplace=True),
            )
        if use_proprio:
            # Low dimensional and already meaningful: no encoder beyond an MLP.
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, proprio_feature), nn.ReLU(inplace=True),
            )
        if use_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(goal_dim, goal_feature), nn.ReLU(inplace=True),
            )
        if use_language:
            if language_mode == "tokens":
                self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=mmc.PAD)
                self.language_encoder = nn.Sequential(
                    nn.Linear(embed_dim, language_feature), nn.ReLU(inplace=True),
                )
            else:
                # L0: one embedding per task. No word order exists here, so test T2
                # (shuffled instruction) is UNDEFINED for this arm -- which is precisely
                # why the language channel uses tokens instead.
                self.task_embedding = nn.Embedding(num_tasks, language_feature)

        dims = []
        if use_image:
            dims.append(image_feature)
        if use_proprio:
            dims.append(proprio_feature)
        if use_goal:
            dims.append(goal_feature)
        if use_language:
            dims.append(language_feature)
        self.fusion_dims = dims

        self.fusion = nn.Sequential(
            nn.Linear(sum(dims), fusion_hidden), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, fusion_out), nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(fusion_out, horizon * action_dim)

    def encode_language(self, ids):
        """Masked mean pool over token embeddings.

        Mean pooling is permutation-invariant, so this encoder cannot see word
        order. That is what makes test T2 (shuffled instruction, lesson 3.8.4.6) a
        *negative control on the implementation* rather than a test of language
        use: T2 must leave the action unchanged, and if it moves, either this
        pooling is not what ran or positional information entered somewhere.
        """
        x = self.token_embedding(ids)                                 # [B, T, E]
        keep = (ids != mmc.PAD).unsqueeze(-1).to(x.dtype)             # [B, T, 1]
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)       # [B, E]
        return self.language_encoder(pooled)

    def forward(self, batch):
        parts = []
        if self.flags["use_image"]:
            img = batch["image"]
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            if img.dim() == 4 and img.shape[-1] == 3:                 # HWC -> CHW
                img = img.permute(0, 3, 1, 2).contiguous()
            parts.append(self.image_encoder(img))
        if self.flags["use_proprio"]:
            parts.append(self.proprio_encoder(batch["proprio"]))
        if self.flags["use_goal"]:
            parts.append(self.goal_encoder(batch["goal"]))
        if self.flags["use_language"]:
            if self.language_mode == "tokens":
                parts.append(self.encode_language(batch["language_ids"]))
            else:
                parts.append(self.task_embedding(batch["task_index"]))

        z = self.fusion(torch.cat(parts, dim=-1))
        return self.action_head(z).view(-1, self.horizon, self.action_dim)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trace_shapes(model, batch, device=None):
    """Forward once and report the output shape of every parameterised submodule.

    The point is to be able to read the information path off the notebook output
    instead of trusting the constructor.
    """
    device = device or next(model.parameters()).device
    rows, handles = [], []

    def make(name):
        def fn(mod, inp, out):
            rows.append((name, type(mod).__name__,
                         tuple(out.shape) if isinstance(out, torch.Tensor) else "-"))
        return fn

    for name, mod in model.named_modules():
        if name and any(True for _ in mod.parameters(recurse=False)):
            handles.append(mod.register_forward_hook(make(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(to_device(batch, device))
    model.train(was_training)
    for h in handles:
        h.remove()
    return rows


# --------------------------------------------------------------------------- #
# Batching and training (CPU-sized: this dataset has 637 samples in total)
# --------------------------------------------------------------------------- #
class DictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mask=None):
        self.dataset = dataset
        self.idx = (np.arange(len(dataset["start_of"])) if mask is None
                    else np.flatnonzero(mask))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        out = {}
        for k in FIELDS:
            v = np.asarray(self.dataset[k][j])
            # np.ascontiguousarray promotes a 0-d scalar to shape (1,), which would make
            # task_index [B, 1] and the embedding output [B, 1, F]; torch.cat of a 3-d and
            # a 2-d part then fails. Keep scalar fields 0-d so collate yields [B].
            out[k] = torch.from_numpy(v if v.ndim == 0 else np.ascontiguousarray(v))
        return out


def to_device(batch, device=DEVICE):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_model(model, train_loader, val_loader, epochs=150, lr=1e-3, weight_decay=0.0,
                seed=0, verbose_every=25, device=DEVICE):
    """Adam on MSE, best-validation checkpointing, and no silent broadcasting.

    ``F.mse_loss`` broadcasts silently, so a ``[B, 8, 8]`` prediction against a
    ``[B, 8]`` target becomes a ``[B, B, 8]`` garbage loss with no error. The shape
    assertion is the guard, not the documentation.
    """
    model = model.to(device)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = dict(val=float("inf"), epoch=-1, state=None)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            yb = batch["action_chunk"]
            pred = model(batch)
            assert pred.shape == yb.shape, f"pred {tuple(pred.shape)} vs target {tuple(yb.shape)}"
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item() * len(yb)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                yb = batch["action_chunk"]
                pred = model(batch)
                assert pred.shape == yb.shape, f"pred {tuple(pred.shape)} vs target {tuple(yb.shape)}"
                va += F.mse_loss(pred, yb).item() * len(yb)

        tr /= len(train_loader.dataset)
        va /= len(val_loader.dataset)
        history.append((epoch, tr, va))
        if va < best["val"]:
            best = dict(val=va, epoch=epoch,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}  train {tr:.5f}  val {va:.5f}"
                  f"   {'*' if best['epoch'] == epoch else ''}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return dict(best_val=best["val"], best_epoch=best["epoch"], history=history)


@torch.no_grad()
def predict(model, dataset, mask=None, horizon=None, batch_size=64, device=DEVICE):
    """Model predictions and targets for the masked samples, in NORMALISED units.

    ``horizon`` slices the chunk axis; ``None`` keeps all H rows. Comparing a single
    row against the whole chunk is not meaningful (lesson 3.7: interiors of a long
    and a short chunk are averages over different numbers of terms), so callers must
    say which they want.
    """
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(dataset["start_of"]))
    assert len(idx) > 0, "no samples selected"
    model.eval()
    preds, targets = [], []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        b = to_device({k: torch.from_numpy(np.ascontiguousarray(dataset[k][chunk]))
                       for k in FIELDS}, device)
        p = model(b).cpu()
        preds.append(p if horizon is None else p[:, horizon, :])
        t = b["action_chunk"].cpu()
        targets.append(t if horizon is None else t[:, horizon, :])
    return torch.cat(preds, 0), torch.cat(targets, 0)


def mse(pred, target):
    assert pred.shape == target.shape, f"{tuple(pred.shape)} vs {tuple(target.shape)}"
    return float(torch.nn.functional.mse_loss(pred, target))


def gripper_sign_accuracy(pred, target, dim=7):
    """Fraction of samples whose predicted gripper SIGN matches the expert's."""
    assert pred.shape == target.shape and pred.dim() == 2, tuple(pred.shape)
    return float(((pred[:, dim] > 0) == (target[:, dim] > 0)).float().mean())


# --------------------------------------------------------------------------- #
# Counterfactual evaluation (lesson 3.8.4.6)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def instruction_sensitivity(model, dataset, mask, mode, env_id=None,
                            horizon=0, batch_size=64, device=DEVICE):
    """``|| pi(o, l_correct) - pi(o, l_mode) ||`` over the masked samples.

    Offline loss cannot answer whether the model uses the instruction: replacing
    the instruction does not change the target, so the loss necessarily rises and
    carries no information. Sensitivity is a function of the model's output.

    ``horizon=0`` restricts the comparison to the first chunk row, which is the
    only fair single-step quantity (lesson 3.7: interiors of chunks are averages
    over different numbers of terms).

    Returns ``(mean_abs, per_dim_abs, mean_signed)`` in NORMALISED action units.
    """
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(dataset["start_of"]))
    if env_id is not None:
        idx = idx[dataset["task_of"][idx] == env_id]
    assert len(idx) > 0, "no samples selected"
    if mode not in ("shuffled", "contradictory", "drop"):
        raise ValueError(f"unknown mode {mode!r}")

    # Only two instructions exist, so resolve each task's replacement ids once.
    by_task = {t: mmc.instruction_variants(t) for t in sorted(set(dataset["task_of"]))}

    model.eval()
    diffs = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        base = {k: torch.from_numpy(np.ascontiguousarray(dataset[k][chunk])) for k in FIELDS}
        alt = dict(base)
        ids = base["language_ids"].clone()
        if mode == "drop":
            ids[:] = mmc.PAD
        else:
            for r, j in enumerate(chunk):
                ids[r] = torch.from_numpy(by_task[dataset["task_of"][j]][mode])
        alt["language_ids"] = ids

        a = model(to_device(base, device))[:, horizon, :]
        b = model(to_device(alt, device))[:, horizon, :]
        diffs.append((b - a).cpu())
    d = torch.cat(diffs, 0)
    return d.abs().mean().item(), d.abs().mean(0).numpy(), d.mean(0).numpy()
