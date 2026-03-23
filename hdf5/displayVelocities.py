import h5py
import matplotlib.pyplot as plt
import numpy as np

#PATH = "/mnt/data/IsolineCode/mixing_extract_uni/mixing_T10/out10/tracers.h5"
#PATH = "/media/jorge/Crucial X9/NewH5/tracers_f12.h5"

PATH = "/media/jorge/Crucial X9/NewH5/velocities_f12.h5"

#  time_list = [   2    4    6    8   10   12   14   16   18   20   22   24   26   28
#   32   34   38   42   46   50   54   60   66   72   80   86   96  104
#  114  126  138  150  166  182  198  218  238  262  286  314  344  378
#  414  454  498  546  598  656  718  788  864  946 1038 1138 1246 1366
# 1498 1642 1800 1954 2122 2302 2500]

#Datasets available:
#['Muv', 'Muv_cut', 'Ufx', 'Ufx_cut', 'Ufy', 'Ufy_cut']
#File attributes:
#  crop_end_1based_x = 2144
#  crop_start_1based_x = 358
#  dx_m = 8.399e-06
#  dy_m = 8.399e-06
#  time_list = [   2    4    6    8   10   12   14   16   18   20   22   24   26   28
#   32   34   38   42   46   50   54   60   66   72   80   86   96  104
#  114  126  138  150  166  182  198  218  238  262  286  314  344  378
#  414  454  498  546  598  656  718  788  864  946 1038 1138 1246 1366
# 1498 1642 1800 1954 2122 2302 2500]

#Datasets available:
#['C_cut_T10', 'C_cut_T1038', 'C_cut_T104', 'C_cut_T1138', 'C_cut_T114', 'C_cut_T12', 'C_cut_T1246', 'C_cut_T126', 'C_cut_T1366', 'C_cut_T138', 'C_cut_T14', 'C_cut_T1498', 'C_cut_T15>
#File attributes:
#  crop_end_1based_x = 2144
#  crop_start_1based_x = 358
#  dx_m = 8.399e-06
#  dy_m = 8.399e-06
#  time_list = [   2    4    6    8   10   12   14   16   18   20   22   24   26   28
#   32   34   38   42   46   50   54   60   66   72   80   86   96  104
#  114  126  138  150  166  182  198  218  238  262  286  314  344  378
#  414  454  498  546  598  656  718  788  864  946 1038 1138 1246 1366
# 1498 1642 1800 1954 2122 2302 2500]



#time = 138
dset_name = 'Ufx_cut'
dset_nameY = 'Ufy_cut'

with h5py.File(PATH, "r") as f:
    if dset_name not in f:
        raise KeyError(f"{dset_name} not found. Available: {list(f.keys())[:10]} ...")
    
    X = f[dset_name][:]
    
    Y = f[dset_nameY][:]
    dx = f.attrs["dx_m"]
    dy = f.attrs["dy_m"]

# build physical coordinates (so axes are in meters)
Ny, Nx = X.shape
x_coords = np.arange(Nx) * dx
y_coords = np.arange(Ny) * dy


C = np.sqrt(X**2+Y**2)


plt.figure(figsize=(8,6))
pcm = plt.pcolormesh(x_coords, y_coords, C,
                     shading="auto", cmap="turbo", vmin=0, vmax=1)
plt.colorbar(pcm, label="Tracer concentration")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Velocity Field Magnitude")
plt.axis("equal")
plt.show()
