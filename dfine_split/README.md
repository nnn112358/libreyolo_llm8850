# dfine_split — D-FINE 精度優先プリセット（NPU backbone + CPU FP32 decoder）

U16量子化で D-FINE は decoder が劣化する（held-out 13枚で float基準 recall 84%）。
**backbone+encoder を NPU(U16)、DETR decoder を CPU FP32 に分割**すると recall 90% に回復する。
原因と全測定は [../ACCURACY.md](../ACCURACY.md)。

```
images ──> dfine_backbone.axmodel (NPU, U16) ──2特徴マップ(FP32)──> dfine_decoder.onnx (CPU, FP32) ──> pred_logits/pred_boxes
            backbone+encoder：CNN中心・量子化に強い            DETR decoder：TopK/Gather/LayerNorm/Softmax・量子化に弱い
```

## 構成ファイル

| ファイル | 役割 | 実行先 |
|---|---|---|
| `dfine_backbone.axmodel` | backbone+encoder（COCO128再キャリブ U16） | NPU (axengine) |
| `dfine_decoder.onnx` | DETR decoder | CPU (onnxruntime, FP32) |
| `infer_dfine_hybrid.py` | 単一ファイル完結のハイブリッド推論 | — |

## 使い方

```bash
pip install onnxruntime           # axengine / opencv-python / numpy に加えて必要
python infer_dfine_hybrid.py                       # 同梱 sample を推論
python infer_dfine_hybrid.py --image foo.jpg --conf 0.4
```

## 精度・速度（held-out 13枚 / float ONNX 基準, AX650N実測）

| 構成 | recall | meanIoU | scoreMAE | レイテンシ |
|---|--:|--:|--:|--:|
| 全NPU U16（`../dfine.axmodel`） | 84–85% | 0.94 | 0.03–0.04 | ~25 ms |
| **本ハイブリッド（分割）** | **90.0%** | **0.95** | **0.025** | ~31 ms |

速度優先なら全NPU（`../infer_dfine.py`）、精度優先なら本プリセット。
decoder は `graph_optimization_level=ALL` + `intra_op_num_threads=4` で最適化済み。

## 再現

`dfine_backbone.onnx` / `dfine_decoder.onnx` は元の `dfine_n.onnx` を
encoder→decoder の2特徴マップ（`fpn_blocks.0/cv4` 40×40, `pan_blocks.0/cv4` 20×20）で
`onnx.utils.extract_model` 分割し、backbone のみ pulsar2(U16, COCO128キャリブ)で axmodel 化したもの。
float同士の合成は元モデルを完全再現（max|Δ|=0）。
