from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path

# ---------- pick a crop box interactively ----------
def pick_crop_box(img_path):
    img = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Drag to select crop. Close the window when done.")

    crop_box = {"box": None}  # (left, upper, right, lower)

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        left, right = sorted([int(round(x1)), int(round(x2))])
        upper, lower = sorted([int(round(y1)), int(round(y2))])
        crop_box["box"] = (left, upper, right, lower)
        print("Selected crop box:", crop_box["box"])

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],          # left mouse button
        interactive=True
    )

    plt.show()

    if crop_box["box"] is None:
        raise RuntimeError("No crop selected (did you close the window without dragging a box?)")

    return crop_box["box"]

# ---------- apply crop box to any image ----------
def crop_with_box(img_path, out_path, box):
    img = Image.open(img_path).convert("RGB")
    cropped = img.crop(box)
    cropped.save(out_path, quality=95)
    return cropped.size

# ---------- EDIT THESE ----------

dir = Path.cwd()


img1 = dir / "Glow_Flow_April13.png"
img2 = dir / "Glow_Flow_April13_NF.png"
out1 = dir / "Glow_Flow_April13_crop.png"
out2 = dir /"Glow_Flow_April13_NF_crop.png"

# 1) choose crop on first image
box = pick_crop_box(img1)

# 2) apply same crop to both
size1 = crop_with_box(img1, out1, box)
size2 = crop_with_box(img2, out2, box)

print("Done.")
print("Saved:", out1, "size:", size1)
print("Saved:", out2, "size:", size2)
print("Crop box used:", box)
