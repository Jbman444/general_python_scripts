import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm

PATH = "/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10/out10/tracers.h5"
OUT_DIR = Path("/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10/out10/tracer_images_log")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with h5py.File(PATH, "r") as f:
    times = list(np.array(f.attrs["time_list"], dtype=int))  # authoritative list
    dx = f.attrs["dx_m"]
    dy = f.attrs["dy_m"]

    for i, t in enumerate(times, 1):
        dset_name = f"C_cut_T{t}"
        if dset_name not in f:
            print(f"[{i}/{len(times)}] Missing {dset_name}, skipping")
            continue

        C = f[dset_name][:]
        Ny, Nx = C.shape
        x_coords = np.arange(Nx) * dx
        y_coords = np.arange(Ny) * dy

        # mask NaNs and values <= 0 (log scale can’t handle 0 or negative)
        C_masked = np.ma.masked_where((C <= 0) | np.isnan(C), C)

        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(
            x_coords, y_coords, C_masked,
            shading="auto", cmap="turbo",
            norm=LogNorm(vmin=1e-3, vmax=1.0)  # adjust vmin as needed
        )
        fig.colorbar(pcm, ax=ax, label="Tracer concentration (log scale)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Tracer field at time {t}")
        ax.set_aspect("equal")

        out_path = OUT_DIR / f"tracer_T{t}_log.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

        print(f"[{i}/{len(times)}] Saved {out_path}")
