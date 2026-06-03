# libreyolo_llm8850

[LibreYOLO](https://github.com/LibreYOLO/libreyolo) からエクスポートした物体検出モデルを、
**AX650N NPU（M.2 アクセラレータ「LLM8850」など）** 上で
[axengine](https://github.com/AXERA-TECH/pyaxengine)（`AXCLRTExecutionProvider`）を使って推論する、
単一ファイル完結の最小スクリプト集です。

ONNX を `pulsar2 6.0` で **U16 量子化**して `.axmodel` に変換済みのモデルを同梱しており、
`axengine` + `OpenCV` + `NumPy` だけで NPU 推論が動きます。

## 同梱モデル

| スクリプト | ファミリー | モデル | 入力 | 前処理 | 後処理 |
|---|---|---|---|---|---|
| `infer_yolo9.py` | YOLOv9 | `yolo9.axmodel` | 640 | letterbox(左上詰め, pad=114) + RGB + /255 | conf → **NMS** → 逆スケール |
| `infer_dfine.py` | D-FINE | `dfine.axmodel` | 640 | 単純リサイズ + RGB + /255 | DETR 集合予測 **(NMS フリー)** |
| `infer_rfdetr.py` | RF-DETR | `rfdetr.axmodel` | 384 | 単純リサイズ + RGB + /255 + ImageNet 正規化 | DETR 集合予測 (NMS フリー / COCO91→80) |

> `.axmodel` は ONNX のメタデータ（クラス名 / `imgsz`）を持たないため、クラス名は COCO80 を
> スクリプトに内蔵し、`imgsz` は入力テンソルの shape から読み取ります。

## 動作環境

- **AX650N NPU を搭載したデバイス**（M.2 アクセラレータ「LLM8850」など）。`axengine` の
  `AXCLRTExecutionProvider` は AXCL ランタイム（`libaxcl_rt` 等）を呼ぶため、AX のボード／ホスト環境が必要です。
- Python 3.8+ / NumPy / OpenCV

## セットアップ

`axengine` ランタイム本体はこのリポジトリには含めていません。AX650N デバイス上で別途入手してください
（[pyaxengine](https://github.com/AXERA-TECH/pyaxengine) のリリース、またはデバイスの BSP に同梱のもの）。

```bash
# 推論スクリプトが import する依存(numpy / opencv-python)
pip install -r requirements.txt
# axengine は別途インストール(例)
#   pip install <pyaxengine の wheel>
```

## 使い方

引数なしで実行すると、同梱の `sample_640x480.jpg` を推論し、結果画像を
`sample_640x480_<family>_axmodel.jpg` として入力画像と同じ場所に保存します。

```bash
python infer_yolo9.py
python infer_dfine.py
python infer_rfdetr.py
```

任意の画像・しきい値を指定する場合:

```bash
python infer_yolo9.py --model yolo9.axmodel --image foo.jpg --conf 0.3 --iou 0.45
python infer_dfine.py  --model dfine.axmodel  --image foo.jpg --conf 0.4
python infer_rfdetr.py --model rfdetr.axmodel --image foo.jpg --conf 0.4
```

| オプション | 説明 | 既定 |
|---|---|---|
| `--model` | 使用する `.axmodel` | 各スクリプトの同名モデル |
| `--image` | 入力画像 | `sample_640x480.jpg` |
| `--out`   | 出力画像パス | `<入力>_<family>_axmodel.jpg` |
| `--conf`  | 信頼度しきい値 | `0.25` |
| `--iou`   | NMS の IoU しきい値（yolo9 のみ） | `0.45` |

## 推論結果（サンプル）

`sample_640x480.jpg` に対する各モデルの NPU 推論結果（`image/` に同梱）:

| YOLOv9 | D-FINE | RF-DETR |
|---|---|---|
| ![yolo9](image/sample_640x480_yolo9_axmodel.jpg) | ![dfine](image/sample_640x480_dfine_axmodel.jpg) | ![rfdetr](image/sample_640x480_rfdetr_axmodel.jpg) |

## 量子化精度に関する注意（U16）
同梱の `.axmodel` は `pulsar2 6.0` / target=**AX650** / **U16 量子化** で変換しています。

| モデル | 出力 cos | 判定 |
|---|--:|---|
| yolo9 | **0.99995** | ✅ ほぼ無劣化（CNN ベースは U16 に強い） |
| dfine | **0.814** | ⚠️ 明確な劣化（DETR の集合予測・LayerNorm が量子化感度高） |
| rfdetr | **0.812** | ⚠️ 明確な劣化（DINOv2 系 transformer + attention） |


> `axengine` ランタイム本体は含めていません（[動作環境](#動作環境)・[セットアップ](#セットアップ)参照）。

## ライセンス

本リポジトリのコードは [MIT License](LICENSE)。
モデルの元 OSS は [LibreYOLO](https://github.com/LibreYOLO/libreyolo)（MIT）、
ランタイムは [axengine / pyaxengine](https://github.com/AXERA-TECH/pyaxengine)（BSD）です。
