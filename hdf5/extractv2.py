# -*- coding: utf-8 -*-
"""
CSV -> gridded tracer/velocity fields (SAVED AS (y,x)), crops, HDF5 saves,
and (optional) PNG previews.

Conventional orientation:
    arrays.shape = (Ny, Nx)  # axis-0 = y (rows), axis-1 = x (cols)
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------
# User settings
# ------------------------
data_dir = Path("/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10")      # <- adjust
out_dir  = Path("/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10/out10") # <- adjust
DO_VELOCITY = True   # build+save velocity fields
DO_PNG      = True   # save PNG previews

time_vec = [
    2, 4, 6 , 8, 10, 12, 14, 16, 18,
    20, 22, 24, 26, 28, 32, 34, 38, 42, 46, 50, 54,
    60, 66, 72, 80, 86, 96, 104, 114, 126, 138, 150, 166,
    182, 198, 218, 238, 262, 286, 314, 344, 378, 414, 454, 498,
    546, 598, 656, 718, 788, 864, 946, 1038, 1138, 1246, 1366, 1498, 1642, 1800
]

dx = dy = 8.399e-6
x_shift = 0.003
crop_start_1based = 358   # MATLAB inclusive (x direction)
crop_end_1based   = 2144  # MATLAB inclusive (x direction)

vel_h5 = out_dir / "velocities.h5"
trc_h5 = out_dir / "tracers.h5"

# ------------------------
# Helpers
# ------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def csv_to_array(filename: Path) -> np.ndarray:
    df = pd.read_csv(filename)
    return df.values.astype(float)

def to_grid_indices(x_m, y_m, dxy):
    # 1-based like MATLAB, then convert to 0-based for numpy
    ix1 = np.rint((x_m + dxy) / dxy).astype(int)
    iy1 = np.rint((y_m + dxy) / dxy).astype(int)
    return ix1 - 1, iy1 - 1   # 0-based

def make_custom_colormap():
    anchor = np.array([
        [0.96, 0.96, 0.86],  # Beige
        [0.00, 0.50, 0.00],  # Green
        [0.00, 0.00, 0.00],  # Black
    ])
    return LinearSegmentedColormap.from_list("beige_green_black", anchor, N=256)

def h5_write(h5f: h5py.File, dset_name: str, arr: np.ndarray, **attrs):
    if dset_name in h5f:
        del h5f[dset_name]
    dset = h5f.create_dataset(
        dset_name, data=arr,
        compression="gzip", compression_opts=4, shuffle=True, fletcher32=True
    )
    for k, v in attrs.items():
        dset.attrs[k] = v

# ------------------------
# Main
# ------------------------
ensure_dir(out_dir)
cmap = make_custom_colormap()

# overwrite files cleanly
vel_f = h5py.File(vel_h5, "w")
trc_f = h5py.File(trc_h5, "w")

# file-level metadata
for f in (trc_f, vel_f):
    f.attrs["dx_m"] = dx
    f.attrs["dy_m"] = dy
    f.attrs["crop_start_1based_x"] = crop_start_1based
    f.attrs["crop_end_1based_x"]   = crop_end_1based
trc_f.attrs["time_list"] = np.array(time_vec, dtype=np.int32)
vel_f.attrs["time_list"] = np.array(time_vec, dtype=np.int32)

velocity_done = False
s = crop_start_1based - 1         # to 0-based
e = crop_end_1based               # end-exclusive slice keeps 2144 inclusive

try:
    for t in time_vec:
        flow = csv_to_array(data_dir / f"T_t{t}.csv")
        # columns: 0:Time, 1:x, 2:y, 3:z, 4:T, 5:Ux, 6:Uy, 7:Uz
        x, y, z = flow[:, 1], flow[:, 2], flow[:, 3]
        Tvals, Ux, Uy = flow[:, 4], flow[:, 5], flow[:, 6]

        # preprocess
        x = x + x_shift
        ix0, iy0 = to_grid_indices(x, y, dx)
        keep = z <= 0.0
        ix0, iy0 = ix0[keep], iy0[keep]
        Tvals, Ux, Uy = Tvals[keep], Ux[keep], Uy[keep]

        # grid extents (0-based -> +1)
        Nx = int(ix0.max()) + 1
        Ny = int(iy0.max()) + 1

        # -------- Velocity fields (first time only), saved as (Ny, Nx) --------
        if DO_VELOCITY and not velocity_done:
            Ufx = np.zeros((Ny, Nx), dtype=float)  # (y,x)
            Ufy = np.zeros((Ny, Nx), dtype=float)
            Ufx[iy0, ix0] = Ux
            Ufy[iy0, ix0] = Uy
            Muv = np.sqrt(Ufx**2 + Ufy**2)

            h5_write(vel_f, "Ufx", Ufx, axes="(y,x)")
            h5_write(vel_f, "Ufy", Ufy, axes="(y,x)")
            h5_write(vel_f, "Muv", Muv, axes="(y,x)")

            # crop in x-direction -> slice COLUMNS
            Ufx_cut = Ufx[:, s:e]
            Ufy_cut = Ufy[:, s:e]
            Muv_cut = Muv[:, s:e]
            h5_write(vel_f, "Ufx_cut", Ufx_cut, crop="x:358-2144 (MATLAB)", axes="(y,x)")
            h5_write(vel_f, "Ufy_cut", Ufy_cut, crop="x:358-2144 (MATLAB)", axes="(y,x)")
            h5_write(vel_f, "Muv_cut", Muv_cut, crop="x:358-2144 (MATLAB)", axes="(y,x)")

            velocity_done = True

        # -------- Tracer field, saved as (Ny, Nx) --------
        C = np.full((Ny, Nx), np.nan, dtype=float)
        C[iy0, ix0] = Tvals

        # crop in x-direction -> slice columns
        C_cut = C[:, s:e]
        porosity = (C_cut.size - np.isnan(C_cut).sum()) / C_cut.size
        h5_write(trc_f, f"C_cut_T{t}", C_cut, porosity=porosity, time=int(t), axes="(y,x)")

        # -------- PNG preview (no transpose needed now) --------
        if DO_PNG:
            Ny_cut, Nx_cut = C_cut.shape
            x_coords = np.arange(Nx_cut + 1) * dx  # horizontal
            y_coords = np.arange(Ny_cut + 1) * dy  # vertical

            fig, ax = plt.subplots(figsize=(8.5, 7.0))
            ax.set_facecolor((0.7, 0.7, 0.7))  # gray for NaNs/background

            pcm = ax.pcolormesh(x_coords, y_coords, C_cut,
                                shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)

            ax.contour(x_coords[:-1], y_coords[:-1], C_cut,
                       levels=[0.5], colors='r', linewidths=2.0)

            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Tracer T")
            fig.savefig(out_dir / f"Time{t}.png", dpi=600,
                        bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)

finally:
    trc_f.close()
    vel_f.close()

print(f"Done. Saved (y,x) arrays to: {out_dir}")
