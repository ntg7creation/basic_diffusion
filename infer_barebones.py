# infer_barebones.py
import os
import json
import re
import argparse
import yaml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from models.unet_cond2d import UNetCond2D
from diffusion.ddpm import DDPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--ckpt", required=True, help="Checkpoint model path")
    ap.add_argument("--text", required=True, help="Prompt text")
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--outdir", default=None, help="Where to save results (default = <exp_dir>/infer)")
    args = ap.parse_args()

    # ---- Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Default output directory: <experiment>/infer
    if args.outdir is None:
        exp_dir = os.path.dirname(os.path.dirname(args.ckpt))  # e.g. save/barebones
        outdir = os.path.join(exp_dir, "infer")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    device = cfg["device"]

    # ---- Load model checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    unet = UNetCond2D(
        in_channels=1,
        cond_channels=cfg["cond_channels"],
        base_channels=64,
        time_dim=128,
        out_channels=1,
    ).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    text_proj = torch.nn.Linear(768, cfg["cond_channels"]).to(device)  # DistilBERT hidden=768
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    ddpm = DDPM(timesteps=cfg["timesteps"]).to(device)

    # ---- Text encoder
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

    # ---- Build cond map
    texts = [args.text] * args.num_samples
    with torch.no_grad():
        tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        last = text_encoder(**tok).last_hidden_state
        mask = tok.get("attention_mask", None)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = last[:, 0]
        cond_vec = text_proj(pooled)  # [B, Ccond]

    B = args.num_samples
    H, W = cfg["frames"], cfg.get("dmvb", 126)
    cond_map = cond_vec.view(B, cfg["cond_channels"], 1, 1).expand(B, cfg["cond_channels"], H, W)

    # ---- Sample
    with torch.no_grad():
        imgs = ddpm.sample(unet, (B, 1, H, W), cond_map, device, progress=True)  # [-1,1]

    # ---- Convert back to motion and (optionally) denorm to original scale
    motions = imgs.squeeze(1).cpu().numpy()                 # [B, H, W] in [-1,1]
    motions = np.clip(motions, -1.0, 1.0) * cfg["rescale_to_unit"]  # undo scale

    if "mean_std_dir" in cfg and cfg["mean_std_dir"]:
        mean = np.load(os.path.join(cfg["mean_std_dir"], f'mean_{cfg["side"]}.npy')).astype(np.float32)  # [D]
        std  = np.load(os.path.join(cfg["mean_std_dir"], f'std_{cfg["side"]}.npy')).astype(np.float32)   # [D]
        for i in range(B):
            z = motions[i]                                   # [H, W] = [T, D]
            z = np.clip(z, -10, 10)                          # safety
            z = z * std + mean
            motions[i] = z

    # ---- Save each motion separately as JSONL
    n_joints = W // 3
    for i in range(B):
        # reshape [H, W] -> [H, n_joints, 3]
        motion = motions[i].reshape(H, n_joints, 3)
        # add opacity column
        opacity = np.ones((H, n_joints, 1), dtype=np.float32)
        motion = np.concatenate([motion, opacity], axis=-1)  # [H, n_joints, 4]

        # sanitize text for filename
        safe_text = re.sub(r'[^a-zA-Z0-9]+', '_', texts[i].strip().lower())
        filename = f"motion_{i:03d}_{safe_text}.jsonl"
        out_path = os.path.join(outdir, filename)

        # save as JSONL: one line = one frame
        with open(out_path, "w") as f:
            for frame in motion:
                f.write(json.dumps(frame.tolist()) + "\n")

        print(f"✔ Saved motion {i} → {out_path}")



if __name__ == "__main__":
    main()
