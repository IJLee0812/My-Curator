"""DeepStream YOLOE ONNX export (detection, open-vocabulary).

Exports YOLOE (seg or detect weights) with text embeddings fused into the
detector head for DeepStream nvinfer. Classes are chosen at export time via
``--custom-classes``; no runtime text prompts needed.

After ``head.fuse(txt_pe)``, YOLOE reduces to a standard Detect forward —
same output tensor layout as YOLO26, so the existing
``libnvdsinfer_custom_impl_Yolo.so`` parser works as-is.

Usage (inside container):
    python3 scripts/export_yoloe.py \\
        -w models/yoloe-26m-seg.pt \\
        --custom-classes "vehicle,person,motorcycle,traffic_sign" \\
        --dynamic --simplify
    # → models/yoloe-26m-seg.onnx  +  models/labels.txt

The exported ``.onnx`` and generated ``labels.txt`` are placed next to the
weights file. Point ``configs/config_infer_yolo26e.txt`` at those paths.
"""

import os
import sys
from copy import deepcopy

import onnx
import torch
import torch.nn as nn
import ultralytics.models.yolo
import ultralytics.utils
import ultralytics.utils.tal as _m
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, C3k2, Detect

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


class YOLOEDetectWrapper(nn.Module):
    """Strip mask coefficients from fused YOLOE so output is detection-only.

    Seg-weights output [B, 4+nc+mask_coeffs, N]; detect-weights output
    [B, 4+nc, N]. This wrapper uniformly returns [B, 4+nc, N] so the
    downstream ``DeepStreamOutput`` formats bbox+score+label consistently.
    """

    def __init__(self, model, nc):
        super().__init__()
        self.model = model
        self.nc = nc

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, : 4 + self.nc, :]


def yoloe_export(weights, device, custom_classes, fuse=False):
    model = YOLO(weights)
    model.set_classes(custom_classes)
    nc = len(custom_classes)

    txt_pe = model.model.get_text_pe(custom_classes)
    head = model.model.model[-1]
    head.fuse(txt_pe)

    inner_model = deepcopy(model.model).to(device)
    for p in inner_model.parameters():
        p.requires_grad = False
    inner_model.eval()
    inner_model.float()

    # NOTE: Do NOT call inner_model.fuse() here — on modern Ultralytics
    # (≥8.4.38) that path invokes ``YOLOEDetect.fuse(txt_feats=None)``
    # which nulls cv2/cv3/cv4, breaking later forward pass with
    # KeyError: 'feats'. Conv+BN fusion is a nice-to-have optimization
    # and can be skipped safely; the already-fused text embedding is
    # what matters for export.
    if fuse:
        inner_model = inner_model.fuse()

    for _, m in inner_model.named_modules():
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            m.end2end = False
        elif isinstance(m, (C2f, C3k2)):
            m.forward = m.forward_split

    wrapped = YOLOEDetectWrapper(inner_model, nc)
    wrapped.eval()
    return wrapped, custom_classes


def main(args):
    import warnings

    warnings.filterwarnings("ignore")

    print(f"\nStarting: {args.weights}")
    custom_classes = [item.strip() for item in args.custom_classes.split(",")]

    device = torch.device("cpu")
    model, names = yoloe_export(args.weights, device, custom_classes)

    # Write labels next to the exported ONNX (not bare "labels.txt")
    # so YOLOE doesn't clobber the legacy COCO-80 labels.txt used by YOLO26.
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"
    labels_file = args.weights.rsplit(".", 1)[0] + ".labels.txt"
    print(f"Creating {labels_file} ({len(names)} classes)")
    with open(labels_file, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")

    model = nn.Sequential(model, DeepStreamOutput())
    img_size = args.size * 2 if len(args.size) == 1 else args.size
    onnx_input = torch.zeros(args.batch, 3, *img_size).to(device)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print("Exporting to ONNX")
    torch.onnx.export(
        model,
        onnx_input,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None,
        dynamo=False,  # Ultralytics models fail with dynamo exporter on torch>=2.5
    )

    if args.simplify:
        print("Simplifying ONNX")
        import onnxslim

        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}")
    print(f"→ Set num-detected-classes={len(names)} in your nvinfer config\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream YOLOE ONNX export (detection)")
    parser.add_argument("-w", "--weights", required=True, type=str)
    parser.add_argument(
        "--custom-classes",
        required=True,
        type=str,
        help='Comma-separated class names, e.g. "vehicle,person,motorcycle"',
    )
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set --dynamic and --batch>1 at the same time")

    main(args)
