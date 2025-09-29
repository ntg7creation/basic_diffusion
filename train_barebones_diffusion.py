# train_barebones_diffusion.py
import os
import argparse
import yaml
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from datasets.gigahands_image import GigaHandsImageT2M
from models.unet_cond2d import UNetCond2D
from diffusion.ddpm import DDPM


def pool_text_embeddings(last_hidden_state, attention_mask=None, strategy="cls"):
    """
    Convert token-level embeddings to a single vector per sample.
    - "cls": take hidden[0] (DistilBERT doesn't have true CLS token but first token works)
    - "mean": mean over tokens (mask-aware)
    """
    if strategy == "mean":
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
    return last_hidden_state[:, 0]  # [B, hidden]


def text_to_cond_map(tokenizer, text_encoder, texts, H, W, cond_channels=8, device="cuda"):
    """
    texts -> DistilBERT -> pooled vector [B, hidden]
    """
    with torch.no_grad():
        tok = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)
        out = text_encoder(**tok).last_hidden_state  # [B, seq, hidden]
        pooled = pool_text_embeddings(out, tok.get("attention_mask", None), strategy="mean")
    return pooled  # [B, hidden]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    # Load yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)


    # ---- Pretty print config
    print("=" * 80)
    print("🚀 Training Config")
    for k, v in cfg.items():
        print(f"{k:20s}: {v}")
    print("=" * 80)


    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ---- Dataset
    ds = GigaHandsImageT2M(
        root_dir=cfg["root_dir"],
        annotation_file=cfg["annotation_file"],
        mean_std_dir=cfg["mean_std_dir"],
        split="train",
        side=cfg["side"],
        fixed_len=cfg["frames"],
        rescale_to_unit=cfg["rescale_to_unit"],
    )
    

    # safer worker choice
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4  # or cfg.get("num_workers", 4)

    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=num_workers, drop_last=True)

    # ---- Text encoder
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(cfg["device"]).eval()

    # Find hidden_dim of BERT
    with torch.no_grad():
        tmp_tok = tokenizer(["tmp"], return_tensors="pt").to(cfg["device"])
        hidden_dim = text_encoder(**tmp_tok).last_hidden_state.shape[-1]

    # Projector from BERT hidden -> cond_channels
    text_proj = nn.Linear(hidden_dim, cfg["cond_channels"]).to(cfg["device"])

    # ---- Model & Diffusion
    unet = UNetCond2D(
        in_channels=1,
        cond_channels=cfg["cond_channels"],
        base_channels=64,
        time_dim=128,
        out_channels=1,
    ).to(cfg["device"])
    ddpm = DDPM(timesteps=cfg["timesteps"]).to(cfg["device"])

    # ---- Optim
    opt = torch.optim.AdamW(list(unet.parameters()) + list(text_proj.parameters()), lr=cfg["lr"])

    # ---- Train
    global_step = 0
    unet.train()
    for epoch in range(cfg["epochs"]):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        for imgs, texts in pbar:
            imgs = imgs.to(cfg["device"])  # [B,1,H,W]
            B, _, H, W = imgs.shape

            with torch.no_grad():
                pooled = text_to_cond_map(tokenizer, text_encoder, texts, H, W,
                                          cfg["cond_channels"], cfg["device"])

            cond_vec = text_proj(pooled)  # [B, Ccond]
            cond_map = cond_vec.view(B, cfg["cond_channels"], 1, 1).expand(B, cfg["cond_channels"], H, W)

            t = torch.randint(0, cfg["timesteps"], (B,), device=cfg["device"], dtype=torch.long)

            loss = ddpm.loss(unet, imgs, t, cond_map)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if global_step % cfg["save_interval"] == 0:
                ckpt = {
                    "unet": unet.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "cfg": cfg,
                    "step": global_step,
                    "epoch": epoch,
                }
                torch.save(ckpt, os.path.join(cfg["save_dir"], f"model_{global_step:08d}.pt"))

        # save each epoch too
        ckpt = {
            "unet": unet.state_dict(),
            "text_proj": text_proj.state_dict(),
            "cfg": cfg,
            "step": global_step,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], f"model_epoch_{epoch+1}.pt"))


if __name__ == "__main__":
    main()
