import h5py
import matplotlib.pyplot as plt
import numpy as np

#PATH = "/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10/out10/tracers.h5"
PATH = "/media/jorge/Crucial X9/NewH5/tracers_f12.h5"


#  time_list = [   2    4    6    8   10   12   14   16   18   20   22   24   26   28
#   32   34   38   42   46   50   54   60   66   72   80   86   96  104
#  114  126  138  150  166  182  198  218  238  262  286  314  344  378
#  414  454  498  546  598  656  718  788  864  946 1038 1138 1246 1366
# 1498 1642 1800 1954 2122 2302 2500]



time = 138
dset_name = f"C_cut_T{time}"

with h5py.File(PATH, "r") as f:
    if dset_name not in f:
        raise KeyError(f"{dset_name} not found. Available: {list(f.keys())[:10]} ...")
    
    C = f[dset_name][:]
    dx = f.attrs["dx_m"]
    dy = f.attrs["dy_m"]

# build physical coordinates (so axes are in meters)
Ny, Nx = C.shape
x_coords = np.arange(Nx) * dx
y_coords = np.arange(Ny) * dy

plt.figure(figsize=(8,6))
pcm = plt.pcolormesh(x_coords, y_coords, C,
                     shading="auto", cmap="turbo", vmin=0, vmax=1)
plt.colorbar(pcm, label="Tracer concentration")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Tracer field at time {time}")
plt.axis("equal")
plt.show()
