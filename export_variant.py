import argparse
import pathlib

import numpy as np
import tensorflow as tf

from train import balanced_calibration, load, predict_tflite

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="output_variants")
ap.add_argument("--calib_per_class", type=int, default=50)
a = ap.parse_args()

root, out = pathlib.Path(a.data), pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(a.model)
train_raw = load(root / "train", 32, True)
test = load(root / "test", 32, False)
names = train_raw.class_names
calib = balanced_calibration(train_raw, len(names), a.calib_per_class)


def convert(kind):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if kind == "float32":
        return conv.convert()
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if kind == "float16":
        conv.target_spec.supported_types = [tf.float16]
    if kind == "int8_full":
        conv.representative_dataset = lambda: ([x] for x in calib)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return conv.convert()


healthy = [i for i, n in enumerate(names) if "healthy" in n.lower()]
rows = []
for kind in ["float32", "float16", "dynamic_range", "int8_full"]:
    data = convert(kind)
    path = out / f"model_{kind}.tflite"
    path.write_bytes(data)
    yt, yp = predict_tflite(path, test)
    acc = (yt == yp).mean()
    if healthy:
        h = np.isin(yt, healthy)
        h_rec = np.isin(yp[h], healthy).mean() if h.any() else float("nan")
        sick = ~h
        danger = int(np.isin(yp[sick], healthy).sum())
        rows.append((kind, len(data) / 1e3, acc, h_rec, danger, int(sick.sum())))
    else:
        rows.append((kind, len(data) / 1e3, acc, float("nan"), 0, len(yt)))

print(f"\n{'variant':<15}{'size KB':>9}{'accuracy':>10}{'healthy recall':>16}{'sick->healthy':>15}")
for kind, kb, acc, hr, d, n in rows:
    print(f"{kind:<15}{kb:>9.0f}{acc:>10.3f}{hr:>16.3f}{d:>9} of {n}")
