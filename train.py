import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

IMG = 224
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- data
def load(path, batch, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=(IMG, IMG), batch_size=batch,
        shuffle=shuffle, seed=42, label_mode="int")


def augment(images, labels):
    """Random changes that imitate field conditions. Applied to TRAINING data only."""
    images = tf.cast(images, tf.float32)
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, 50.0)            # dim or bright light
    images = tf.image.random_contrast(images, 0.6, 1.4)          # glare / dull light
    images = images + tf.random.normal(tf.shape(images), 0, 6.0)  # cheap-camera noise
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def class_weights_from(ds, n_classes, power=1.0):
    """Give small classes a bigger say, so 'healthy' is not ignored.
    power=1.0 is full balancing, 0.5 is gentler, 0 turns weighting off."""
    counts = np.zeros(n_classes)
    for _, y in ds:
        for c in y.numpy():
            counts[c] += 1
    total = counts.sum()
    weights = {i: float((total / (n_classes * counts[i])) ** power) for i in range(n_classes)}
    return counts, weights


# --------------------------------------------------------------- model
def build_model(n_classes, weights):
    # include_preprocessing=True: the model takes raw 0..255 pixels itself,
    # so the phone app does not need to normalise anything.
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG, IMG, 3), include_top=False, weights=weights,
        minimalistic=True, include_preprocessing=True)
    base.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# ------------------------------------------------------------- export
def balanced_calibration(ds, n_classes, per_class):
    picked = {c: [] for c in range(n_classes)}
    for imgs, labels in ds:
        for k in range(imgs.shape[0]):
            c = int(labels[k])
            if len(picked[c]) < per_class:
                picked[c].append(imgs[k:k + 1].numpy().astype(np.float32))
        if all(len(v) >= per_class for v in picked.values()):
            break
    flat = [x for v in picked.values() for x in v]
    print("Calibration images per class:", {c: len(v) for c, v in picked.items()})
    return flat


def export_int8(model, calib_images, out_path):
    def representative():
        for x in calib_images:
            yield [x]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    data = conv.convert()
    pathlib.Path(out_path).write_bytes(data)
    print(f"Saved {out_path}  ({len(data) / 1e3:.0f} KB)")


def predict_tflite(path, ds):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    y_true, y_pred = [], []
    for imgs, labels in ds:
        for i in range(imgs.shape[0]):
            x = imgs[i:i + 1].numpy().astype(np.float32)
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            y_pred.append(int(np.argmax(interp.get_tensor(out["index"]))))
            y_true.append(int(labels[i]))
    return np.array(y_true), np.array(y_pred)


# ------------------------------------------------------------- report
def make_report(y_true, y_pred, names, title):
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    healthy = [i for i, nm in enumerate(names) if "healthy" in nm.lower()]
    disease = [i for i in range(n) if i not in healthy]
    missed = cm[disease][:, healthy].sum() if healthy and disease else 0
    sick = cm[disease].sum() if disease else 0
    lines = [f"=== {title} ===",
             f"Overall accuracy: {(y_true == y_pred).mean():.3f}  ({len(y_true)} images)",
             f"DANGEROUS ERRORS - diseased leaf called healthy: {missed} of {sick} "
             f"({(100 * missed / sick) if sick else 0:.1f}%)",
             "", f"{'class':<30}{'precision':>10}{'recall':>8}{'images':>8}"]
    for i, name in enumerate(names):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0
        rec = tp / cm[i].sum() if cm[i].sum() else 0
        lines.append(f"{name:<30}{prec:>10.3f}{rec:>8.3f}{cm[i].sum():>8}")
    lines += ["", "Confusion matrix (rows = true class, columns = predicted):",
              "  " + "  ".join(f"[{i}]" for i in range(n))]
    for i in range(n):
        lines.append(f"[{i}] {names[i]:<28}" + "  ".join(f"{v:>4}" for v in cm[i]))
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=15)
    ap.add_argument("--lr_ft", type=float, default=1e-4,
                    help="learning rate for the fine-tuning stage")
    ap.add_argument("--unfreeze", type=int, default=30,
                    help="how many top layers of the backbone to fine-tune")
    ap.add_argument("--cw_power", type=float, default=1.0,
                    help="class-weight strength: 1.0 full, 0.5 gentle, 0 off")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--calib_per_class", type=int, default=50)
    ap.add_argument("--weights", default="imagenet",
                    help="'imagenet' (normal) or 'none' (only for quick tests)")
    a = ap.parse_args()

    root, out = pathlib.Path(a.data), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_raw = load(root / "train", a.batch, True)
    val = load(root / "val", a.batch, False).prefetch(AUTOTUNE)
    test = load(root / "test", a.batch, False).prefetch(AUTOTUNE)
    names = train_raw.class_names
    json.dump(names, open(out / "labels.json", "w"), indent=2)
    print("Classes (output order):", names)

    counts, cw = class_weights_from(train_raw, len(names), a.cw_power)
    print("Training images per class:", dict(zip(names, counts.astype(int))))
    print("Class weights:", {names[i]: round(w, 2) for i, w in cw.items()})

    train = train_raw.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    w = None if a.weights == "none" else "imagenet"
    model, base = build_model(len(names), w)

    # Stage 1: train only the new top layer
    compile_model(model, 1e-3)
    stop = lambda: tf.keras.callbacks.EarlyStopping(
        patience=a.patience, restore_best_weights=True)
    model.fit(train, validation_data=val, epochs=a.epochs_head, class_weight=cw,
              callbacks=[stop()])

    if a.epochs_ft > 0:
        base.trainable = True
        for layer in base.layers[:-a.unfreeze]:
            layer.trainable = False
        compile_model(model, a.lr_ft)
        model.fit(train, validation_data=val, epochs=a.epochs_ft, class_weight=cw,
                  callbacks=[stop()])
    model.save(out / "model_float.keras")

    # Float32 results on the test set
    y_true, y_pred = [], []
    for imgs, labels in test:
        y_pred += list(np.argmax(model.predict(imgs, verbose=0), axis=1))
        y_true += list(labels.numpy())
    rep_float = make_report(np.array(y_true), np.array(y_pred), names, "FLOAT32 model, test set")

    # int8 export + same test
    tflite_path = out / "model_int8.tflite"
    calib = balanced_calibration(train_raw, len(names), a.calib_per_class)
    export_int8(model, calib, tflite_path)
    yt, yp = predict_tflite(tflite_path, test)
    rep_int8 = make_report(yt, yp, names, "INT8 TFLite model, test set")

    settings = ("SETTINGS: " + ", ".join(f"{k}={v}" for k, v in vars(a).items()
                                         if k not in ("data", "out")))
    report = settings + "\n\n" + rep_float + "\n\n" + rep_int8
    report += ("\n\nNOTE: this test set is lab-style PlantVillage data. "
               "Expect lower accuracy on real field photos.")
    (out / "report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
