import os
from PIL import Image

IMG_SIZE = 112

root = "docs\\lfw\\"

for subdir, dirs, files in os.walk(root):
    out_path = os.path.basename(subdir) + ".bin"
    for filename in files:
        if filename.lower().endswith(".jpg"):
            in_path = os.path.join(subdir, filename)
            with Image.open(in_path) as img:
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                imf = img.convert("RGB")

                pixels = img.load()

                with open("docs\\" + out_path, "ab") as f:
                    for i in range(3):
                        for j in range(IMG_SIZE):
                            for k in range(IMG_SIZE):
                                f.write((pixels[k, j][i]).to_bytes(1))
    print(f"Saved {out_path}")
