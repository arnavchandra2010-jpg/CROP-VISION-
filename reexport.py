import argparse
import pathlib

import numpy as np
import tensorflow as tf

from train import (balanced_calibration, export_int8, load, make_report,
                   predict_tflite)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="output_reexport")
ap.add_argument("--calib_per_class", type=int, default=50)
a = ap.parse_args()

root, out = pathlib.Path(a.data), pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(a.model)
train_raw = load(root / "train", 32, True)
test = load(root / "test", 32, False)
names = train_raw.class_names

calib = balanced_calibration(train_raw, len(names), a.calib_per_class)
path = out / "model_int8.tflite"
export_int8(model, calib, path)

yt, yp = predict_tflite(path, test)
report = make_report(yt, yp, names, "INT8 TFLite model, balanced calibration, test set")
(out / "report.txt").write_text(report)
print("\n" + report)
