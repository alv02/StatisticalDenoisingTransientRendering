import os
import shutil
import subprocess
import sys

import mitsuba as mi
import numpy as np

# ============================================================
# CONFIG
# ============================================================
if len(sys.argv) != 4:
    print("Usage: python script.py <VIDEO_PATH> <GBUFFER_PATH> <OUT_PATH>")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
GBUFFER_EXR = sys.argv[2]
OUT_PATH = sys.argv[3]

TMP_DIR = "tmp_oidn"


# ============================================================
# PFM UTILITIES (OIDN CLI REQUIRES PFM)
# ============================================================
def save_pfm(path, img):
    img = np.flipud(img)
    H, W, C = img.shape
    assert C == 3

    with open(path, "wb") as f:
        f.write(b"PF\n")
        f.write(f"{W} {H}\n".encode())
        f.write(b"-1.0\n")  # little endian
        img.astype(np.float32).tofile(f)


def load_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        assert header == "PF"

        W, H = map(int, f.readline().decode().split())
        scale = float(f.readline().decode())
        data = np.fromfile(f, dtype=np.float32)

    img = data.reshape((H, W, 3))
    return np.flipud(img)


# ============================================================
# LOAD DATA
# ============================================================
mi.set_variant("cuda_ad_rgb")

video = np.load(VIDEO_PATH).astype(np.float32)
H, W, T, C = video.shape
assert C == 3

bitmap = mi.Bitmap(GBUFFER_EXR)
res = dict(bitmap.split())

albedo = np.array(res["albedo"], dtype=np.float32)
normals = np.array(res["nn"], dtype=np.float32)
albedo = np.zeros_like(albedo)
normals = np.zeros_like(normals)


# ============================================================
# NORMALIZE NORMALS
# ============================================================
n = np.linalg.norm(normals, axis=-1, keepdims=True)
normals = normals / np.clip(n, 1e-6, None)

# ============================================================
# PREP TMP FILES
# ============================================================
os.makedirs(TMP_DIR, exist_ok=True)

alb_pfm = os.path.join(TMP_DIR, "albedo.pfm")
nrm_pfm = os.path.join(TMP_DIR, "normal.pfm")

save_pfm(alb_pfm, albedo)
save_pfm(nrm_pfm, normals)

# ============================================================
# DENOISE LOOP (OIDN CLI)
# ============================================================
denoised_video = np.empty_like(video)

for t in range(T):
    color_pfm = os.path.join(TMP_DIR, f"color_{t:04d}.pfm")
    out_pfm = os.path.join(TMP_DIR, f"denoised_{t:04d}.pfm")

    save_pfm(color_pfm, video[:, :, t, :])

    subprocess.run(
        [
            "oidnDenoise",
            "--hdr",
            color_pfm,
            "--alb",
            alb_pfm,
            "--nrm",
            nrm_pfm,
            "--clean_aux",
            "--output",
            out_pfm,
        ],
        check=True,
    )

    denoised_video[:, :, t, :] = load_pfm(out_pfm)

# ============================================================
# SAVE RESULT
# ============================================================
np.save(OUT_PATH, denoised_video)

print("✔ OIDN finished correctly:", OUT_PATH)

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    print(f"✔ Temporal folder '{TMP_DIR}' deleted")
