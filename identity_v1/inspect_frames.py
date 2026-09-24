import cv2
from pathlib import Path
from PIL import Image, ImageDraw

cases = {
    "Cosmos2.5_0016":
        "/shared/ssd_30T/zhuoyingyang/physact/cosmos_predict25/native_outputs/robowm_68_full/0016.mp4",

    "Cosmos3_seed101_0003":
        "/shared/ssd_30T/zhuoyingyang/physact/cosmos3/export_robowm68_3seeds/seed101/0003.mp4",

    "LVP_0004":
        "/shared/ssd_30T/zhuoyingyang/physact/evaluation_results/robowm_lvp_68_numbered/videos/0004.mp4",
}

outdir = Path("inspection")
outdir.mkdir(exist_ok=True)

fractions = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

for case, path in cases.items():

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    inds = [
        min(total - 1, round((total - 1) * f))
        for f in fractions
    ]

    imgs = []

    for idx in inds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        img.thumbnail((420, 260))

        tile = Image.new("RGB", (420, 300), "black")
        x = (420 - img.width) // 2
        tile.paste(img, (x, 30))

        d = ImageDraw.Draw(tile)
        d.text((10, 8), f"{case} | frame {idx}/{total-1}", fill="white")

        imgs.append(tile)

    cap.release()

    canvas = Image.new(
        "RGB",
        (420 * 3, 300 * 2),
        "black"
    )

    for k, img in enumerate(imgs):
        canvas.paste(
            img,
            ((k % 3) * 420, (k // 3) * 300)
        )

    out = outdir / f"{case}.jpg"
    canvas.save(out, quality=95)

    print(out)

