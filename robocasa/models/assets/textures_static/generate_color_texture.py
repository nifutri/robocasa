from PIL import Image
import os


colors = {"white": (255, 255, 255), "dark_green": (18, 68, 41)}


for c_name in colors:
    orig_path = c_name + ".png"
    rot_path = os.path.join("rotated", c_name + "_rot.png")

    im = Image.new("RGB", (256, 256), colors[c_name])
    im.save(orig_path)
    im.save(rot_path)
