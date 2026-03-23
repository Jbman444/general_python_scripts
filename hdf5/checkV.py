import h5py

with h5py.File("/media/jorge/Crucial X9/NewH5/velocities_f12.h5", "r") as f:
    print("Datasets available:")
    print(list(f.keys()))       # list all dataset names
    print("File attributes:")
    for k, v in f.attrs.items():
        print(f"  {k} = {v}")
