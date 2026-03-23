import numpy as np
import h5py
import matplotlib.pyplot as plt

# =======================
# TUNABLE PARAMETERS
# =======================
FIGSIZE = (8, 6)

# Colorbar spacing/size (works with fig.colorbar)
CB_PAD      = 0.004   # smaller = closer gap to axes (in fraction of axes)
CB_FRACTION = 0.035   # colorbar width vs. axes width
CB_SHRINK   = 0.58    # colorbar length vs. axes height
CB_ASPECT   = 18      # larger = thinner bar
CB_ANCHOR   = (0.5, 0.5)  # center vertically

# =======================
# DATA
# =======================
PATH = "/media/jorge/Crucial X9/NewH5/velocities_f12.h5"
dset_x, dset_y = "Ufx_cut", "Ufy_cut"

with h5py.File(PATH, "r") as f:
    Ux = f[dset_x][:]
    Uy = f[dset_y][:]
    dx = f.attrs["dx_m"]; dy = f.attrs["dy_m"]

Ny, Nx = Ux.shape
x = np.arange(Nx) * dx
y = np.arange(Ny) * dy

U = np.hypot(Ux, Uy)

# Use percentiles to avoid zeros dominating vmin
nz = U[U > 0]
vmin = np.nanpercentile(nz, 1) if nz.size else 0.0
vmax = np.nanpercentile(nz, 99) if nz.size else 1.0

# Mask zeros and set color for masked values
U_plot = np.ma.masked_where(U == 0, U)
cmap = plt.cm.viridis.copy()
cmap.set_bad('gray')

# Cell-edge coords so there’s no padding
x_edges = np.r_[x[0]-dx/2, (x[:-1]+x[1:])/2, x[-1]+dx/2]
y_edges = np.r_[y[0]-dy/2, (y[:-1]+y[1:])/2, y[-1]+dy/2]

# =======================
# PLOT
# =======================
fig, ax = plt.subplots(figsize=FIGSIZE)

# pcm = ax.pcolormesh(x_edges, y_edges, U_plot, shading="auto",
#                     cmap=cmap, vmin=vmin, vmax=vmax)

# scale in cm 
pcm = ax.pcolormesh(x_edges, y_edges, U_plot*100, shading="auto",
                    cmap=cmap, vmin=vmin*100, vmax=vmax*100)

#ax.set_xlabel("x [m]")
#ax.set_ylabel("y [m]")

ax.set_xlabel("")
ax.set_ylabel("")


# Move x-axis label/ticks to the top (optional)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.tick_params(axis='x', top=True, bottom=False)
ax.spines['bottom'].set_visible(False)

# Loosen right margin to let fig.colorbar position the bar close to the axes
fig.subplots_adjust(right=0.95)

# ---- Colorbar attached to ax ----
cbar = fig.colorbar(
    pcm, ax=ax, location="right", orientation="vertical",
    pad=CB_PAD, fraction=CB_FRACTION, shrink=CB_SHRINK,
    aspect=CB_ASPECT, anchor=CB_ANCHOR
)
# cbar.set_label("|U| [m/s]")
cbar.set_label("|U| [cm/s]")


# ---- Remove axis numbers & frame (do NOT call axis('off')) ----
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Exact bounds, no extra margins
ax.set_xlim(x_edges[0], x_edges[-1])
ax.set_ylim(y_edges[0], y_edges[-1])
ax.set_aspect("equal", adjustable="box")
ax.margins(0)

fig.savefig("/media/jorge/Crucial X9/newFigures/T12_velocityfield_v1.png",
            dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.show()
