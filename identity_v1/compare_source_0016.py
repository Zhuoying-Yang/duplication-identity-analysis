import cv2
from PIL import Image, ImageDraw
from pathlib import Path

source_path = "/shared/ssd_30T/zhuoyingyang/physact/evaluation_results/robowm_lvp_68_numbered/images/0016.png"

video_path = "/shared/ssd_30T/zhuoyingyang/physact/cosmos_predict25/native_outputs/robowm_68_full/0016.mp4"

out = Path("inspection/source_vs_cosmos25_0016.jpg")
out.parent.mkdir(exist_ok=True)

W, H = 420, 300

def tile(img, label):
    img = img.copy()
    img.thumbnail((W, H-35))

    x = Image.new("RGB", (W, H), "black")
    xx = (W-img.width)//2
    x.paste(img, (xx, 35))

    d = ImageDraw.Draw(x)
    d.text((8, 8), label, fill="white")
    return x


# source
src = Image.open(source_path).convert("RGB")
tiles = [tile(src, "SOURCE IMAGE")]


# generated frames
cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
    idx = round((total-1)*frac)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()

    if not ok:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    tiles.append(
        tile(img, f"GENERATED frame {idx}")
    )

cap.release()


canvas = Image.new(
    "RGB",
    (W*3, H*2),
    "black"
)

for k, t in enumerate(tiles):
    canvas.paste(
        t,
        ((k%3)*W, (k//3)*H)
    )

canvas.save(out, quality=95)

print(out)
