"""Check Stage 7 checkpoint structure."""
import torch

ckpt_path = r"C:\IsaacLab\logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Top-level keys:", list(ckpt.keys()))
for key in ckpt:
    val = ckpt[key]
    if isinstance(val, dict):
        print(f"\n  {key} (dict, {len(val)} keys):")
        for k2 in sorted(val.keys()):
            v2 = val[k2]
            if hasattr(v2, 'shape'):
                print(f"    {k2}: {v2.shape}")
            else:
                print(f"    {k2}: {v2}")
    elif hasattr(val, 'shape'):
        print(f"  {key}: {val.shape}")
    else:
        print(f"  {key}: {val}")
