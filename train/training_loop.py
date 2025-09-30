# train/training_loop.py
import os
import torch
import logging
from tqdm import tqdm
from train_barebones_diffusion import text_to_cond_vec


def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),  # new log per run
            logging.StreamHandler()                   # also print to console
        ]
    )
    logging.info("📜 Logging initialized")
    logging.info(f"Logs will be saved to {log_path}")


def run_loop(cfg, model, text_proj, tokenizer, text_encoder, ddpm, dataloader, save_dir):
    setup_logger(save_dir)

    opt = torch.optim.AdamW(list(model.parameters()) + list(text_proj.parameters()), lr=cfg["lr"])

    global_step = 0
    model.train()

    for epoch in range(cfg["epochs"]):
        logging.info(f"🚀 Starting epoch {epoch+1}/{cfg['epochs']}")
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        
        epoch_loss = 0.0
        num_batches = 0

        for imgs, texts in pbar:
            imgs = imgs.to(cfg["device"]).squeeze(1)  # [B,H,W]
            B, H, W = imgs.shape

            with torch.no_grad():
                pooled = text_to_cond_vec(tokenizer, text_encoder, texts, cfg["device"])
            cond_vec = text_proj(pooled)

            t = torch.randint(0, cfg["timesteps"], (B,), device=cfg["device"], dtype=torch.long)
            loss = ddpm.loss(model, imgs, t, cond_vec)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            num_batches += 1
            epoch_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            # log every 100 steps
            if global_step % 100 == 0:
                logging.info(f"[Step {global_step}] Loss = {loss.item():.6f}")

            if global_step % cfg["save_interval"] == 0:
                ckpt_path = os.path.join(save_dir, f"model_{global_step:08d}.pt")
                ckpt = {
                    "model": model.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "cfg": cfg,
                    "step": global_step,
                    "epoch": epoch,
                }
                torch.save(ckpt, ckpt_path)
                logging.info(f"💾 Saved checkpoint → {ckpt_path}")

        # compute average loss for this epoch
        avg_loss = epoch_loss / max(1, num_batches)

        # save each epoch too
        ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        ckpt = {
            "model": model.state_dict(),
            "text_proj": text_proj.state_dict(),
            "cfg": cfg,
            "step": global_step,
            "epoch": epoch,
        }
        torch.save(ckpt, ckpt_path)
        logging.info(f"💾 Saved epoch checkpoint → {ckpt_path}")
        logging.info(f"✅ Finished epoch {epoch+1}/{cfg['epochs']} | Avg Loss = {avg_loss:.6f}")
# train/training_loop.py
import os
import torch
import logging
from tqdm import tqdm
from train_barebones_diffusion import text_to_cond_vec


def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),  # new log per run
            logging.StreamHandler()                   # also print to console
        ]
    )
    logging.info("📜 Logging initialized")
    logging.info(f"Logs will be saved to {log_path}")


def run_loop(cfg, model, text_proj, tokenizer, text_encoder, ddpm, dataloader, save_dir):
    setup_logger(save_dir)

    opt = torch.optim.AdamW(list(model.parameters()) + list(text_proj.parameters()), lr=cfg["lr"])

    global_step = 0
    model.train()

    for epoch in range(cfg["epochs"]):
        logging.info(f"🚀 Starting epoch {epoch+1}/{cfg['epochs']}")
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        
        epoch_loss = 0.0
        num_batches = 0

        for imgs, texts in pbar:
            imgs = imgs.to(cfg["device"]).squeeze(1)  # [B,H,W]
            B, H, W = imgs.shape

            with torch.no_grad():
                pooled = text_to_cond_vec(tokenizer, text_encoder, texts, cfg["device"])
            cond_vec = text_proj(pooled)

            t = torch.randint(0, cfg["timesteps"], (B,), device=cfg["device"], dtype=torch.long)
            loss = ddpm.loss(model, imgs, t, cond_vec)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            num_batches += 1
            epoch_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            # log every 100 steps
            if global_step % 100 == 0:
                logging.info(f"[Step {global_step}] Loss = {loss.item():.6f}")

            if global_step % cfg["save_interval"] == 0:
                ckpt_path = os.path.join(save_dir, f"model_{global_step:08d}.pt")
                ckpt = {
                    "model": model.state_dict(),
                    "text_proj": text_proj.state_dict(),
                    "cfg": cfg,
                    "step": global_step,
                    "epoch": epoch,
                }
                torch.save(ckpt, ckpt_path)
                logging.info(f"💾 Saved checkpoint → {ckpt_path}")

        # compute average loss for this epoch
        avg_loss = epoch_loss / max(1, num_batches)

        # save each epoch too
        ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        ckpt = {
            "model": model.state_dict(),
            "text_proj": text_proj.state_dict(),
            "cfg": cfg,
            "step": global_step,
            "epoch": epoch,
        }
        torch.save(ckpt, ckpt_path)
        logging.info(f"💾 Saved epoch checkpoint → {ckpt_path}")
        logging.info(f"✅ Finished epoch {epoch+1}/{cfg['epochs']} | Avg Loss = {avg_loss:.6f}")
