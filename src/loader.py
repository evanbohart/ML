import sys
from PIL import Image

IMG_SIZE = 32;

img = Image.open("docs\\" + sys.argv[1]);
img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS);
img.convert("RGB");

pixels = img.load();
f = open("docs\\" + sys.argv[2], "wb");

for i in range(3):
    for j in range(IMG_SIZE):
        for k in range(IMG_SIZE):
            f.write((pixels[k, j][i]).to_bytes(1));

f.close();
