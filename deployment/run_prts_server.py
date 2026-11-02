#!/usr/bin/env python3
"""PRTS policy server speaking Isaac-GR00T's ZMQ protocol (tcp://0.0.0.0:5550 / get_action).

PRTS ships only an openpi-style *websocket* server (`scripts/serve_policy.py`, :10000),
but soma's real-robot loop and the tele_humanoid_benchmark sim client both speak the
gr00t REQ/REP wire format. This adapter wraps `prts.interface.PRTSInterface` so a PRTS
checkpoint can be evaluated by exactly the same client as gr00t / fastwam / starVLA --
which is what makes the success rates comparable.

Mirrors starVLA's `deployment/model_server/starvla_gr00t_server.py`; keep the two in
sync if the wire format changes.

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REQ/REP
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config

Action layout (78-D, identical to the other G1 SONIC models):
    action.motion_token(64) + action.left_hand_joints(7) + action.right_hand_joints(7)

Usage:
    python prts_gr00t_server.py --ckpt-path outputs/prts/prts_toilet401/checkpoint-20000 \
        --port 5550 --initial-prompt "walk over to the toilet and stand next to it"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch

MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM      # 78

# The action features, in the order they are concatenated into the 78-D vector. Must match
# `select_action_keys` in the run's configs/post-train/<ds>.yaml.
ACTION_KEYS = (("action.motion_token", MOTION_TOKEN_DIM),
               ("teleop.left_hand_joints", HAND_DIM),
               ("teleop.right_hand_joints", HAND_DIM))


class ActionUnnormalizer:
    """Undo the action normalisation PRTS's own unnormaliser silently skips.

    `prts/interface.py:infer()` ends with

        actions = self.unnormalizer({"action": outputs[..., :action_dim].squeeze(0)})["action"]

    i.e. it looks the stats up under the literal key "action". Our stats are keyed per
    action FEATURE (`action.motion_token`, `teleop.left_hand_joints`,
    `teleop.right_hand_joints`) because that is what `select_action_keys` lists, and
    `Unnormalize.forward` starts each iteration with `if key not in batch: continue`.
    So no key ever matches and the call is an identity: **the server hands the sim
    actions that are still in normalised space.** Nothing errors.

    Upstream does not hit this because LIBERO has a single action feature literally named
    `action`. Two further reasons a key rename alone would not be enough for us:
    `load_normalizer_from_model_dir` classifies features by prefix (`observation*` -> STATE,
    `action*` -> ACTION, everything else `continue`), so both `teleop.*` hand features are
    dropped from `features` entirely and get no stats buffer at all.

    Measured on prts_toilet401 (offline `soma infer-check`, FloorPlan401 ep0), served
    action std vs the demos':

        head                 demo raw std   demo QUANTILE-normalised std   served std
        action.motion_token       0.1138                        0.4145        0.5029
        teleop.right_hand_joints  0.0082                        1.1592        1.0887

    The served numbers track the NORMALISED distribution, which is what pinned this down;
    in the sim it looks like a policy shaking at ~4x the demos' amplitude (applied-std
    0.4168 vs the demos' 0.1187).

    QUANTILE is PRTS's own mode (prts/data/normalize.py), not lerobot's:
        forward : x_norm = 2*(x - q01)/(q99 - q01) - 1
        inverse : x      = (x_norm + 1)/2 * (q99 - q01) + q01
    """

    MODES = ("QUANTILE", "MEAN_STD")

    def __init__(self, lo: np.ndarray, hi: np.ndarray, mode: str):
        self.lo, self.hi, self.mode = lo, hi, mode     # q01/q99, or mean/std

    @classmethod
    def from_stats_file(cls, path, mode: Optional[str] = None) -> Optional["ActionUnnormalizer"]:
        """Build from a checkpoint's statistics.json; None if it cannot be built."""
        import json
        import pathlib
        p = pathlib.Path(path) / "statistics.json"
        if not p.is_file():
            print(f"[prts] no statistics.json in {path}: cannot unnormalize actions", flush=True)
            return None
        payload = json.loads(p.read_text())
        mode = (mode or payload.get("state_mode") or "QUANTILE").upper()
        if mode not in cls.MODES:
            print(f"[prts] statistics.json state_mode={mode!r} unsupported "
                  f"(known: {', '.join(cls.MODES)}); leaving actions normalized", flush=True)
            return None
        names = ("q01", "q99") if mode == "QUANTILE" else ("mean", "std")
        stats = payload.get("stats") or {}
        lo, hi = [], []
        for key, dim in ACTION_KEYS:
            s = stats.get(key) or {}
            if not all(n in s and s[n] is not None for n in names):
                print(f"[prts] statistics.json has no {'/'.join(names)} for {key!r}: "
                      f"leaving actions normalized", flush=True)
                return None
            a, b = (np.asarray(s[n], dtype=np.float32).reshape(-1) for n in names)
            if a.shape[0] != dim or b.shape[0] != dim:
                print(f"[prts] statistics.json {key!r} has dim {a.shape[0]}, expected {dim}: "
                      f"leaving actions normalized", flush=True)
                return None
            lo.append(a)
            hi.append(b)
        return cls(np.concatenate(lo), np.concatenate(hi), mode)

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        if self.mode == "MEAN_STD":
            return actions * self.hi + self.lo
        # A constant dimension has q01 == q99, so the span is 0 and every input maps to that
        # constant -- which is the right answer for it, so no special case is needed.
        return (actions + 1.0) * 0.5 * (self.hi - self.lo) + self.lo


class PRTSPolicy:
    """Adapts PRTSInterface.infer() to the gr00t get_action() contract."""

    def __init__(self, iface, initial_prompt: Optional[str] = None,
                 upsample_factor: int = 1, device: str = "cuda",
                 unnormalizer: Optional[ActionUnnormalizer] = None):
        self.iface = iface
        self._current_prompt = initial_prompt
        self.upsample_factor = max(1, int(upsample_factor))
        self.unnormalizer = unnormalizer
        # Observation tensors must land on the model's device: PRTSInterface's normalizer
        # holds CUDA buffers, so feeding CPU tensors raises "expected all tensors to be on
        # the same device".
        self.device = device

    # ---- gr00t endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "prts gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.monotonic()

        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description'][0][0].")
        self._current_prompt = prompt

        frame = self._extract_image(observation)             # [H,W,3] uint8
        proprio = self._extract_proprio(observation)         # [43] float32 or None

        # PRTSInterface expects LeRobot-style keys with a leading batch dim, images as
        # float CHW in [0,1] -- it runs its own normalizer/processor on top.
        img = (torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
               .unsqueeze(0).to(self.device))
        obs: dict[str, Any] = {
            "observation.images.ego_view": img,
            "task": [prompt],
        }
        if proprio is not None:
            obs["observation.state"] = (torch.from_numpy(proprio).float()
                                        .unsqueeze(0).to(self.device))

        t_inf = time.monotonic()
        with torch.no_grad():
            out = self.iface.infer(obs)
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        # PRTSInterface.infer() returns numpy already, but see the float() patch applied to
        # prts/interface.py: upstream calls .cpu().numpy() on a bfloat16 tensor, which numpy
        # cannot represent ("Got unsupported ScalarType BFloat16").
        raw_actions = out["actions"]
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.detach().float().cpu().numpy()
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] < ACTION_DIM:
            raise RuntimeError(
                f"PRTS returned action dim {actions.shape[-1]}, expected >= {ACTION_DIM}. "
                f"Was the checkpoint trained with --max-action-dim 78?")
        actions = actions[:, :ACTION_DIM]
        if self.unnormalizer is not None:
            actions = self.unnormalizer(actions)

        motion = self._upsample(actions[:, :MOTION_TOKEN_DIM])
        left = self._upsample(actions[:, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_DIM])
        right = self._upsample(actions[:, MOTION_TOKEN_DIM + HAND_DIM:ACTION_DIM])

        print(f"[prts] get_action: T={actions.shape[0]} -> H_out={motion.shape[0]} "
              f"(x{self.upsample_factor}); infer={infer_ms:.0f} ms; prompt={prompt!r}",
              flush=True)

        action_dict = {
            "action.motion_token": np.ascontiguousarray(motion, dtype=np.float32),
            "action.left_hand_joints": np.ascontiguousarray(left, dtype=np.float32),
            "action.right_hand_joints": np.ascontiguousarray(right, dtype=np.float32),
        }
        return action_dict, {"inference_time_s": time.monotonic() - t0,
                             "chunk_horizon": int(motion.shape[0])}

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        v = lang.get("annotation.human.task_description")
        while isinstance(v, (list, tuple, np.ndarray)) and len(v):
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        return v if isinstance(v, str) and v else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        """Newest ego-view frame as HWC uint8.

        The tele sim client sends `img[np.newaxis, np.newaxis]`, i.e. (1, 1, H, W, 3) --
        FIVE dims, under the key "ego_view" (not "video.ego_view"). Real-robot callers may
        send (T, H, W, 3) or bare (H, W, 3). Peel every leading singleton/time axis instead
        of assuming a fixed rank: getting this wrong surfaces far downstream as
        `permute(sparse_coo): input.dim() = 5 is not equal to len(dims) = 3`.
        """
        video = obs.get("video") or {}
        for key in ("video.ego_view", "ego_view", "video.webcam", "webcam"):
            arr = video.get(key) if key in video else obs.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim > 3:          # drop batch/time axes, keeping the newest frame
                arr = arr[-1]
            if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
                raise RuntimeError(f"ego-view image has unexpected shape {np.shape(arr)}; "
                                   f"expected (..., H, W, 3)")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        raise RuntimeError(f"No ego-view image in observation; keys={list(obs)} "
                           f"video={list(video)}")

    # The 43-D G1 state is these joint groups in THIS order (deploy_living_room_vla.py
    # _STATE_SLICES). Do not sort the dict: alphabetical order scrambles the vector, and
    # "projected_gravity" is an EXTRA field that must not be concatenated into the 43.
    STATE_ORDER = ("left_leg", "right_leg", "waist", "left_arm",
                   "left_hand", "right_arm", "right_hand")

    @classmethod
    def _extract_proprio(cls, obs: dict) -> Optional[np.ndarray]:
        state = obs.get("state") or {}
        if not state:
            return None
        keys = ([k for k in cls.STATE_ORDER if k in state]
                if any(k in state for k in cls.STATE_ORDER) else sorted(state))
        parts = []
        for k in keys:
            v = np.asarray(state[k], dtype=np.float32)
            while v.ndim > 1:          # client sends (1, 1, D); keep the newest row
                v = v[-1]
            parts.append(v.ravel())
        return np.concatenate(parts).astype(np.float32) if parts else None

    def _upsample(self, chunk: np.ndarray) -> np.ndarray:
        """Linearly interpolate the action chunk when the control loop wants more steps."""
        if self.upsample_factor == 1 or chunk.shape[0] < 2:
            return chunk
        t_in = np.arange(chunk.shape[0], dtype=np.float32)
        t_out = np.linspace(0, chunk.shape[0] - 1,
                            chunk.shape[0] * self.upsample_factor, dtype=np.float32)
        return np.stack([np.interp(t_out, t_in, chunk[:, d])
                         for d in range(chunk.shape[1])], axis=1).astype(np.float32)


def serve(policy: PRTSPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[prts] PolicyServer ready on tcp://{host}:{port}", flush=True)

    while True:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                sock.send(mnp.packb({"status": "ok"}))
                break
            elif endpoint == "reset":
                resp = policy.reset(data)
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(data.get("observation", data),
                                                 data.get("options"))
                # The sim client does `action, _info = resp[0], resp[1]` -- it expects a
                # LIST, not a dict. Returning the action dict directly makes the client die
                # with a misleading `KeyError: 0` while the server logs look perfectly fine.
                resp = [action, info]
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:                       # keep serving after a bad request
            import traceback
            traceback.print_exc()
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass
    print("[prts] server stopped", flush=True)


def load_policy(args) -> PRTSPolicy:
    if args.prts_repo:
        sys.path.insert(0, args.prts_repo)
    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

    from prts.interface import PRTSInterface
    from prts.models.processing_prts_qwen3_vl import PRTS_Qwen3VLProcessor
    from prts.models.modeling_prts_qwen3_vl import PRTS_Qwen3VL
    from prts.models.configuration_prts_qwen3_vl import PRTS_FlowMatchingConfig_Qwen3VL
    from prts.models.processing_action_tokenizer import UniversalActionProcessor

    ckpt = args.ckpt_path
    processor = PRTS_Qwen3VLProcessor.from_pretrained(ckpt, padding_side="right",
                                                      use_fast=True)
    action_tok = UniversalActionProcessor.from_pretrained(
        f"{os.environ['HF_HUB_CACHE']}/models--physical-intelligence--fast",
        trust_remote_code=True)
    processor.set_action_tokenizer(action_tok)

    config = PRTS_FlowMatchingConfig_Qwen3VL.from_pretrained(ckpt)
    model = PRTS_Qwen3VL.from_pretrained(ckpt, config=config, dtype="auto",
                                         device_map="auto")
    model.set_fast_action_info(processor.action_mapper, processor.action_token_start_index)
    model.eval()

    _align_rope(model, args.rope_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iface = PRTSInterface(
        model=model, processor=processor, model_path=ckpt,
        device=device,
        dtype=torch.bfloat16,
        action_dim=ACTION_DIM,
        image_min_pixel=processor.image_processor.min_pixels,
        image_max_pixel=processor.image_processor.max_pixels,
        state_mode=args.state_mode,
        state_relative_action=args.state_relative,
        # Selects the image patch size get_image_info() resizes with: qwen2_5_vl -> 14
        # (factor 28), qwen3_vl -> 16 (factor 32). MUST be the --rope-type training used,
        # not the model family: see _align_rope().
        model_type=args.rope_type,
    )
    # PRTS's own unnormaliser is a no-op for our multi-key action layout, so do it here.
    # Skip if a future PRTS does key the stats under a literal "action" -- then infer()
    # already unnormalises and doing it twice would be wrong.
    unnorm = None
    if args.unnormalize_actions:
        if "action" in getattr(getattr(iface, "unnormalizer", None), "features", {}):
            print("[prts] PRTSInterface unnormalizer covers 'action': "
                  "leaving unnormalization to it", flush=True)
        else:
            unnorm = ActionUnnormalizer.from_stats_file(ckpt, args.state_mode)
            if unnorm is not None:
                print(f"[prts] unnormalizing actions in the adapter (mode={unnorm.mode}, "
                      f"keys={', '.join(k for k, _ in ACTION_KEYS)})", flush=True)

    print(f"[prts] loaded checkpoint {ckpt} (action_dim={ACTION_DIM}, "
          f"state_mode={args.state_mode}, rope={args.rope_type}, device={device})", flush=True)
    return PRTSPolicy(iface, args.initial_prompt, args.upsample_factor, device,
                      unnormalizer=unnorm)


def _align_rope(model, rope_type: str) -> None:
    """Make inference use the SAME rope index function the checkpoint was trained with.

    PRTS computes vision-token positions in the DATA pipeline and passes them in:
    `prts/data/dataset.py` picks `get_rope_index_3` or `get_rope_index_25` from
    `TrainConfig.rope_type` and puts the result in the batch as `position_ids`. Inference
    does not: `PRTSInterface._process_obs()` builds no `position_ids`, so
    `PRTS_Qwen3VL.sample_actions()` falls back to the model's OWN `get_rope_index` — always
    the Qwen3-VL one.

    That is fine when the two agree, and silently wrong when they don't: `TrainConfig`
    defaults `rope_type` to "qwen2_5_vl" while PRTS-4B is a Qwen3-VL base, so a run that
    forgets the flag (soma's train command used to) trains on Qwen2.5-VL positions and is
    served Qwen3-VL ones — no error, every vision token just sits somewhere it never sat
    during training.

    So --rope-type describes the CHECKPOINT, not the architecture. Read it off the TRAINING
    LOG ("Using Qwen3VL rope index" / "Using Qwen2.5VL rope index"), never off the model
    name. The iThor checkpoints say Qwen3VL, so they need no realignment and qwen3_vl (the
    default) is correct for them.

    WARNING — qwen2_5_vl is not a free "try the other scheme". It also flips PRTSInterface's
    model_type, hence get_image_info()'s pre-resize patch factor 16 -> 14, while the
    checkpoint's preprocessor_config.json keeps patch_size 16: a 240x320 frame becomes
    252x308, the HF fast processor floors the grid to 224x288, and patches.view() raises
    "shape '[1,1,2,3,7,2,16,9,2,16]' is invalid for input of size 465696" (2026-07-29: this
    is what made both PRTS eval cells exit rc=1). Serving a genuinely qwen2_5_vl-trained
    checkpoint would need the processor's patch_size overridden as well; no such checkpoint
    exists here, so that is deliberately not wired up.
    """
    if rope_type == "qwen3_vl":
        return                       # the model's built-in index is already the Qwen3 one
    if rope_type != "qwen2_5_vl":
        raise ValueError(f"--rope-type must be qwen3_vl|qwen2_5_vl, got {rope_type!r}")
    from prts.data.rope2d import get_rope_index_25

    merge = model.config.vision_config.spatial_merge_size
    inner = model.model if hasattr(model, "model") else model

    def _rope25(input_ids, image_grid_thw=None, video_grid_thw=None,
                attention_mask=None, **_):
        return get_rope_index_25(merge, input_ids, image_grid_thw=image_grid_thw,
                                 video_grid_thw=video_grid_thw,
                                 attention_mask=attention_mask)

    for obj in {id(model): model, id(inner): inner}.values():
        obj.get_rope_index = _rope25
    print(f"[prts] rope aligned to TRAINING scheme qwen2_5_vl "
          f"(spatial_merge_size={merge}); the model's built-in Qwen3 index is bypassed",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PRTS policy server (gr00t-compatible, :5550/get_action).")
    p.add_argument("--ckpt-path", required=True,
                   help="PRTS checkpoint dir (e.g. outputs/prts/<run>/checkpoint-20000)")
    p.add_argument("--prts-repo", default=None, help="prepend to sys.path if PRTS isn't installed")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--state-mode", default="QUANTILE",
                   help="must match training (launch_prts.sh uses QUANTILE)")
    p.add_argument("--state-relative", action="store_true",
                   help="set only if trained with state_relative_action=true")
    p.add_argument("--rope-type", default="qwen3_vl", choices=["qwen3_vl", "qwen2_5_vl"],
                   help="the --rope-type the CHECKPOINT was trained with -- read it off the "
                        "TRAINING LOG ('Using Qwen3VL rope index'), not the model name. "
                        "Drives both the rope index function AND the image patch size, so a "
                        "wrong value can crash the image processor (see _align_rope).")
    p.add_argument("--no-unnormalize-actions", dest="unnormalize_actions",
                   action="store_false",
                   help="serve raw model output. Only for debugging: PRTS's own unnormaliser "
                        "silently no-ops on our multi-key action layout, so without this the "
                        "adapter unnormalises from statistics.json (see ActionUnnormalizer).")
    p.add_argument("--upsample-factor", type=int, default=1)
    args = p.parse_args(argv)

    serve(load_policy(args), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
