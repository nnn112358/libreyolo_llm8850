#!/usr/bin/env python3
"""D-FINE ハイブリッド推論（精度優先）: backbone=NPU + decoder=CPU FP32。

U16量子化で劣化するのは DETR decoder（TopK/GatherElements の離散選択 +
LayerNorm/Softmax の誤差増幅と累積）。そこだけ CPU FP32 で動かすと精度が回復する。
backbone(CNN)+encoder は U16 でほぼ無損失(cos 0.997)なので NPU のまま。

  backbone(axengine, dfine_backbone.axmodel) -> 2特徴マップ(FP32)
  decoder (onnxruntime, dfine_decoder.onnx, CPU FP32) -> pred_logits/pred_boxes

前処理・後処理は infer_dfine.py と一致。held-out 13枚で float ONNX 基準 recall
84%(全NPU U16) -> 90%(本ハイブリッド)。詳細は ../ACCURACY.md。

依存: axengine（NPU, 別途入手） + onnxruntime（CPU） + opencv-python + numpy

  python infer_dfine_hybrid.py
  python infer_dfine_hybrid.py --image foo.jpg --conf 0.4
"""
import argparse, os
import cv2, numpy as np
import axengine
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"]


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


def detr_parse(logits, boxes_cxcywh, W, H, conf, max_det=300):
    """DETR集合予測のデコード（NMSなし）。infer_dfine.py と一致。"""
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    Q, nc = prob.shape
    flat = prob.reshape(-1)
    k = min(max_det, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx].astype(np.float32)
    query, cls = idx // nc, idx % nc
    cx, cy, w, h = boxes_cxcywh[query].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xyxy[:, [0, 2]] *= W; xyxy[:, [1, 3]] *= H
    dets = []
    for (x1, y1, x2, y2), sc, c in zip(xyxy, scores, cls):
        if sc <= conf:
            continue
        dets.append((int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H)),
                     int(np.clip(x2, 0, W)), int(np.clip(y2, 0, H)), float(sc), int(c)))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=os.path.join(HERE, "dfine_backbone.axmodel"))
    ap.add_argument("--decoder", default=os.path.join(HERE, "dfine_decoder.onnx"))
    ap.add_argument("--image", default=os.path.join(HERE, "..", "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_dfine_hybrid.jpg"

    bb = axengine.InferenceSession(a.backbone)                 # NPU
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    dec = ort.InferenceSession(a.decoder, sess_options=so, providers=["CPUExecutionProvider"])  # CPU FP32
    binp = bb.get_inputs()[0]
    bout = [o.name for o in bb.get_outputs()]
    onames = [o.name for o in dec.get_outputs()]
    imgsz = int(binp.shape[2])

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"画像を読めません: {a.image}")
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    feats = bb.run(None, {binp.name: blob})                    # NPU backbone -> 2特徴
    outs = dict(zip(onames, dec.run(None, dict(zip(bout, feats)))))  # CPU FP32 decoder
    dets = detr_parse(outs["pred_logits"][0], outs["pred_boxes"][0], W, H, a.conf)

    draw(img, dets); cv2.imwrite(out_path, img)
    print(f"[dfine-hybrid] backbone=NPU decoder=CPU(FP32) imgsz={imgsz} image={W}x{H} detected={len(dets)}")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {NAMES[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
