# train_barebones_diffusion.py
import os
import argparse
import yaml
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from datasets.gigahands_image import GigaHandsImageT2M
from models.transformer_cond import TransformerCond
from diffusion.ddpm import DDPM
from train.training_loop import run_loop
import random
import numpy as np

def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make CUDA deterministic (slows things down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    # --------------------------------- Load yaml -------------------------------- #
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

   # ---------------------------- Pretty print config --------------------------- #
    print("=" * 80)
    print("🚀 Training Config")
    for k, v in cfg.items():
        print(f"{k:20s}: {v}")
    print("=" * 80)

    os.makedirs(cfg["save_dir"], exist_ok=True)

     # ----------------------------------- seed ----------------------------------- #
    set_seed(cfg.get("seed", 42))

    # ---------------------------------- Dataset --------------------------------- #
    ds = GigaHandsImageT2M(
        root_dir=cfg["root_dir"],
        annotation_file=cfg["annotation_file"],
        mean_std_dir=cfg["mean_std_dir"],
        split="train",
        side=cfg["side"],
        fixed_len=cfg["frames"],
        rescale_to_unit=cfg["rescale_to_unit"],
    )
    num_workers = 0 if platform.system() == "Windows" else 4
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=num_workers, drop_last=True)

     # --------------------------- Text encoder (frozen) -------------------------- #
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(cfg["device"]).eval()

    # Find hidden_dim of BERT
    with torch.no_grad():
        tmp_tok = tokenizer(["tmp"], return_tensors="pt").to(cfg["device"])
        hidden_dim = text_encoder(**tmp_tok).last_hidden_state.shape[-1]

    # Projector from BERT hidden -> cond_channels
    text_proj = nn.Linear(hidden_dim, cfg["cond_channels"]).to(cfg["device"])

     # ----------------------------- Model & Diffusion ---------------------------- #
    model = TransformerCond(
        nfeats=cfg["dmvb"],  # feature dimension per frame
        hidden_dim=cfg.get("hidden_dim", 512),
        depth=cfg.get("depth", 8),
        nheads=cfg.get("nheads", 8),
        cond_dim=cfg["cond_channels"]
    ).to(cfg["device"])

    ddpm = DDPM(timesteps=cfg["timesteps"],
                loss_type=cfg.get("loss_type", "mse")).to(cfg["device"])

     # -------------------------------- Train loop -------------------------------- #
    run_loop(cfg, model, text_proj, tokenizer, text_encoder, ddpm, dl, cfg["save_dir"])


if __name__ == "__main__":
    main()
