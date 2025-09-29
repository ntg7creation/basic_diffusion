# datasets/gigahands_image.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join as pjoin
from tqdm import tqdm


class GigaHandsImageT2M(Dataset):
    """
    Minimal dataset → (image, text):
        image: [1, H=frames, W=dmvb] in [-1, 1]
        text:  str
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        mean_std_dir: str,
        split: str = "train",
        side: str = "both",          # 'left', 'right', 'both'
        fixed_len: int = 196,        # frames (H)
        dmvb_layout: str = "full",
        rescale_to_unit: float = 3.0 # divide normalized values by this -> ~[-1,1]
    ):
        
          # ---- Print dataset args
        print("=" * 80)
        print("📦 Initializing GigaHandsImageT2M Dataset")
        print(f" Split            : {split}")
        print(f" Root Dir         : {root_dir}")
        print(f" Annotation File  : {annotation_file}")
        print(f" Mean/Std Dir     : {mean_std_dir}")
        print(f" Side             : {side}")
        print(f" Fixed Length     : {fixed_len} frames")
        print(f" DMVB Layout      : {dmvb_layout}")
        print(f" Rescale to Unit  : {rescale_to_unit}")
        print("=" * 80)

        
        assert side in ("left", "right", "both")
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mean_std_dir = mean_std_dir
        self.side = side
        self.fixed_len = fixed_len
        self.dmvb_layout = dmvb_layout
        self.rescale_to_unit = rescale_to_unit

        mean_path = pjoin(mean_std_dir, f"mean_{side}.npy")
        std_path  = pjoin(mean_std_dir, f"std_{side}.npy")
        self.mean = np.load(mean_path).astype(np.float32)   # [D]
        self.std  = np.load(std_path).astype(np.float32)    # [D]

        # collect (npy_path, text)
        self.samples = []
        with open(self.annotation_file, "r") as f:
            for line in tqdm(f, desc=f"Loading GigaHands [{split}]"):
                ann = json.loads(line)
                scene = ann["scene"]
                seq   = ann["sequence"]
                texts = ann["rewritten_annotation"]
                npy   = pjoin(self.root_dir, scene, "keypoints_3d", seq, f"xyz_{self.side}.npy")
                if os.path.exists(npy):
                    # add multiple text variants for same motion (helps conditioning)
                    for t in texts:
                        self.samples.append((npy, t))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found.")

    def __len__(self):
        return len(self.samples)

    def _normalize_to_image(self, motion: np.ndarray) -> torch.Tensor:
        """
        motion: [T, D]
        1) z-norm per feature using dataset mean/std
        2) rescale to roughly [-1,1] by dividing with rescale_to_unit (default 3)
        3) convert to image [1, H=T, W=D]
        """
        motion = (motion - self.mean) / (self.std + 1e-8)  # z-norm → ~N(0,1)
        motion = np.clip(motion / self.rescale_to_unit, -1.0, 1.0)

        # image layout
        # we want [1, H, W] where H=frames, W=dmvb
        img = torch.from_numpy(motion).float()  # [T, D]
        img = img.unsqueeze(0)                  # [1, T, D]
        return img

    def __getitem__(self, idx):
        npy_path, text = self.samples[idx]
        mot = np.load(npy_path).astype(np.float32)  # [T, D]

        # center-crop / right-pad to fixed_len
        T, D = mot.shape
        if self.fixed_len > 0:
            if T >= self.fixed_len:
                # deterministic crop (can randomize if you prefer)
                start = 0
                mot = mot[start:start + self.fixed_len]
            else:
                # pad by repeating last frame (safer than zeros)
                last = mot[-1][None, :]  # [1, D]
                pad  = np.repeat(last, self.fixed_len - T, axis=0)
                mot  = np.concatenate([mot, pad], axis=0)

        img = self._normalize_to_image(mot)  # [1, H, W]
        return img, text
