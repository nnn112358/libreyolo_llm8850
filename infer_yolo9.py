#!/usr/bin/env python3
"""YOLOv9 .axmodel 推論 (axengine / AX650N)。

family=yolo9。出力 [1, 84, 8400] = (4 xyxy(canvas px) + 80 cls(sigmoid)) x anchors。
前処理: 左上詰め letterbox(pad=114) + RGB + /255 + CHW。
後処理: conf -> NMS(cv2.dnn.NMSBoxes) -> /ratio で元画像座標へ。

参照元: ../onnxruntime/infer_yolo9.py (前処理・後処理は一致, backend のみ axengine)。
axmodel は ONNX メタデータ(クラス名/imgsz)を持たないため、クラス名は COCO80 を内蔵し
imgsz は入力 shape から読む。

  .venv/bin/python infer_yolo9.py
  .venv/bin/python infer_yolo9.py --model yolo9.axmodel --image foo.jpg --conf 0.3
"""
import argparse, os
import cv2, numpy as np
import axengine as ort

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]


def color_for(c):
    np.random.seed(int(c) * 7 + 11)
    return tuple(int(v) for v in np.random.randint(64, 256, size=3))


def draw(img, dets):
    for x1, y1, x2, y2, sc, c in dets:
        col = color_for(c)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{NAMES[c]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), col, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "yolo9.axmodel"))
    ap.add_argument("--image", default=os.path.join(HERE, "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_yolo9_axmodel.jpg"

    sess = ort.InferenceSession(a.model)
    inp = sess.get_inputs()[0]
    imgsz = int(inp.shape[2])                       # NCHW -> H

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"画像を読めません: {a.image}")
    H, W = img.shape[:2]
    # 前処理: 左上詰め letterbox + RGB + /255
    r = min(imgsz / H, imgsz / W)
    nh, nw = int(H * r), int(W * r)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    pred = sess.run(None, {inp.name: blob})[0][0].T     # (8400, 84)
    boxes, scores = pred[:, :4], pred[:, 4:]            # xyxy(canvas px), cls(sigmoid)
    conf = scores.max(1); cls = scores.argmax(1)
    m = conf > a.conf
    boxes, conf, cls = boxes[m], conf[m], cls[m]

    dets = []
    if len(boxes):
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), a.conf, a.iou)
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i] / r               # 元画像座標へ
            dets.append((int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H)),
                         int(np.clip(x2, 0, W)), int(np.clip(y2, 0, H)), float(conf[i]), int(cls[i])))

    draw(img, dets); cv2.imwrite(out_path, img)
    print(f"[yolo9] axmodel={os.path.basename(a.model)} imgsz={imgsz} image={W}x{H} detected={len(dets)}")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {NAMES[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
