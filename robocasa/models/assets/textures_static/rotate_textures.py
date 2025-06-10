from PIL import Image
import os


os.makedirs("rotated", exist_ok=True)

for directory in os.listdir():
    if not os.path.isdir(directory):
        continue

    for image_name in os.listdir(directory):
        if not (image_name.endswith("png") or image_name.endswith("jpg")):
            continue

        new_name = image_name.split(".")[0] + "_rot." + image_name.split(".")[1]
        new_path = os.path.join("rotated", new_name)
        if os.path.exists(new_path):
            continue

        print("Processing:", image_name)
        image = Image.open(os.path.join(directory, image_name))
        image = image.rotate(90)
        image.save(new_path)
