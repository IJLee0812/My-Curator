"""DeepStream YOLOE ONNX export (segmentation, open-vocabulary).

Fuses text embeddings into the detector head and embeds NMS + RoiAlign + mask
decode into the ONNX graph so DeepStream nvinfer (network-type=3) can produce
instance masks directly without post-processing.

Usage (inside container):
    python3 scripts/export_yoloe_seg.py \\
        -w models/yoloe-26m-seg.pt \\
        --custom-classes "vehicle,person" \\
        --dynamic --simplify
    # → models/yoloe-26m-seg-mask.onnx  +  models/labels.txt

Paired with ``configs/config_infer_yolo26e_seg.txt`` and
``lib/libnvdsinfer_custom_impl_Yolo_seg.so``.
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


class RoiAlign(torch.autograd.Function):
    @staticmethod
    def forward(
        self,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode,
        mode,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale,
    ):
        C = X.shape[1]
        num_rois = rois.shape[0]
        return torch.randn(
            [num_rois, C, output_height, output_width], device=rois.device, dtype=rois.dtype
        )

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode,
        mode,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale,
    ):
        return g.op(
            "TRT::ROIAlignX_TRT",
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )


class NMS(torch.autograd.Function):
    @staticmethod
    def forward(self, boxes, scores, score_threshold, iou_threshold, max_output_boxes):
        batch_size = scores.shape[0]
        num_classes = scores.shape[-1]
        num_detections = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        detection_boxes = torch.randn(batch_size, max_output_boxes, 4)
        detection_scores = torch.randn(batch_size, max_output_boxes)
        detection_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        detections_indices = torch.randint(
            0, max_output_boxes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return (
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
            detections_indices,
        )

    @staticmethod
    def symbolic(g, boxes, scores, score_threshold, iou_threshold, max_output_boxes):
        return g.op(
            "TRT::EfficientNMSX_TRT",
            boxes,
            scores,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1,
            score_activation_i=0,
            class_agnostic_i=0,
            box_coding_i=0,
            outputs=5,
        )


class DeepStreamOutput(nn.Module):
    def __init__(self, nc, conf_threshold, iou_threshold, max_detections):
        super().__init__()
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, x):
        preds = x[0].transpose(1, 2)
        boxes = preds[:, :, :4]
        scores = preds[:, :, 4 : self.nc + 4]
        masks = preds[:, :, self.nc + 4 :]
        protos = x[1]

        (
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
            detections_indices,
        ) = NMS.apply(
            boxes,
            scores,
            self.conf_threshold,
            self.iou_threshold,
            self.max_detections,
        )

        batch_size, num_protos, h_protos, w_protos = protos.shape
        total_detections = batch_size * self.max_detections

        batch_index = torch.ones_like(detections_indices) * torch.arange(
            batch_size, device=boxes.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_index = batch_index.view(total_detections).to(torch.int32)
        box_index = detections_indices.view(total_detections).to(torch.int32)

        selected_boxes = boxes[batch_index, box_index]
        selected_masks = masks[batch_index, box_index]

        pooled_proto = RoiAlign.apply(
            protos,
            selected_boxes,
            batch_index,
            1,
            1,
            int(h_protos),
            int(w_protos),
            0,
            0.25,
        )

        masks_protos = torch.matmul(
            selected_masks.unsqueeze(1),
            pooled_proto.view(total_detections, num_protos, h_protos * w_protos),
        )
        masks_protos = masks_protos.sigmoid().view(
            batch_size, self.max_detections, h_protos * w_protos
        )

        return torch.cat(
            [
                detection_boxes,
                detection_scores.unsqueeze(-1),
                detection_classes.unsqueeze(-1),
                masks_protos,
            ],
            dim=-1,
        )


def yoloe_seg_export(weights, device, custom_classes, fuse=False):
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

    # Skip Conv+BN fusion on modern Ultralytics (≥8.4.38): it re-triggers
    # ``YOLOEDetect.fuse(txt_feats=None)`` which nulls cv2/cv3/cv4. See
    # ``export_yoloe.py`` for details.
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

    return inner_model, custom_classes, nc


def main(args):
    import warnings

    warnings.filterwarnings("ignore")

    print(f"\nStarting: {args.weights}")
    custom_classes = [item.strip() for item in args.custom_classes.split(",")]

    device = torch.device("cpu")
    model, names, nc = yoloe_seg_export(args.weights, device, custom_classes)

    # Write labels next to the exported ONNX (not bare "labels.txt")
    # so YOLOE doesn't clobber the legacy COCO-80 labels.txt used by YOLO26.
    base = args.weights.rsplit(".", 1)[0]
    labels_file = base + "-mask.labels.txt"
    print(f"Creating {labels_file} ({len(names)} classes)")
    with open(labels_file, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")

    model = nn.Sequential(
        model,
        DeepStreamOutput(nc, args.conf_threshold, args.iou_threshold, args.max_detections),
    )

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    onnx_input = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = base + "-mask.onnx"

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print("Exporting to ONNX (segmentation)")
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
    print(f"→ Set num-detected-classes={len(names)} in your nvinfer seg config")

    # DS 9.0 / TRT 10 / CUDA 13 workaround: nvinfer's own TRT engine build
    # path aborts with NVRTC_ERROR_COMPILATION when the ONNX embeds the
    # custom EfficientNMSX_TRT / ROIAlignX_TRT ops. ``trtexec`` builds the
    # same engine cleanly, and nvinfer deserializes it without issue.
    engine_path = base + "-mask.engine"
    plugin_path = args.plugin_lib
    trtexec = "/usr/bin/trtexec"
    if args.build_engine and os.path.isfile(plugin_path) and os.path.exists(trtexec):
        import subprocess as _sp

        print(f"Pre-building TRT engine via trtexec → {engine_path}")
        _sp.run(
            [
                trtexec,
                f"--onnx={onnx_output_file}",
                f"--saveEngine={engine_path}",
                "--fp16",
                f"--staticPlugins={plugin_path}",
            ],
            check=True,
        )
        print(f"Done: {engine_path}\n")
    else:
        print(
            "→ Also pre-build the engine so nvinfer skips its broken JIT path:\n"
            f"   trtexec --onnx={onnx_output_file} --saveEngine={engine_path} "
            f"--fp16 --staticPlugins={plugin_path}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepStream YOLOE ONNX export (segmentation)")
    parser.add_argument("-w", "--weights", required=True, type=str)
    parser.add_argument(
        "--custom-classes",
        required=True,
        type=str,
        help='Comma-separated class names, e.g. "vehicle,person"',
    )
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument(
        "--build-engine",
        action="store_true",
        help="Pre-build the TRT engine via trtexec after ONNX export "
        "(recommended on DS 9.0 / TRT 10 — bypasses nvinfer's JIT bug).",
    )
    parser.add_argument(
        "--plugin-lib",
        type=str,
        default="/workspace/lib/libnvdsinfer_custom_impl_Yolo_seg.so",
        help="Path to the custom Yolo seg plugin .so for trtexec "
        "(required when --build-engine is set).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set --dynamic and --batch>1 at the same time")

    main(args)
