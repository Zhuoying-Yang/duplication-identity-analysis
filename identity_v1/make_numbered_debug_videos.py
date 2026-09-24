import cv2
from pathlib import Path

cases = {
    "Cosmos3_seed101_0003":
        "/shared/ssd_30T/zhuoyingyang/physact/cosmos3/export_robowm68_3seeds/seed101/0003.mp4",

    "LVP_0004":
        "/shared/ssd_30T/zhuoyingyang/physact/evaluation_results/robowm_lvp_68_numbered/videos/0004.mp4",
}

outdir = Path("numbered_debug")
outdir.mkdir(exist_ok=True)

for case, path in cases.items():

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outpath = outdir / f"{case}_numbered.mp4"

    writer = cv2.VideoWriter(
        str(outpath),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.rectangle(
            frame,
            (8, 8),
            (230, 58),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"FRAME {idx}",
            (18, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    print(outpath, "frames =", idx)

