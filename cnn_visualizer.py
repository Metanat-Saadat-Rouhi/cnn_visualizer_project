
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32)


def pad_image(image: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")


def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 1) -> np.ndarray:
    kernel = kernel.astype(np.float32)
    image = image.astype(np.float32)
    padded = pad_image(image, padding)
    kh, kw = kernel.shape
    oh = (padded.shape[0] - kh) // stride + 1
    ow = (padded.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)

    for i in range(oh):
        for j in range(ow):
            patch = padded[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def max_pool2d(image: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    h, w = image.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = np.max(patch)
    return out


def get_stage1_kernels() -> list[np.ndarray]:
    return [
        np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]], dtype=np.float32),
        np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]], dtype=np.float32),
        np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]], dtype=np.float32),
        np.array([[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]], dtype=np.float32),
    ]


def get_stage2_kernels() -> list[np.ndarray]:
    return [
        np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=np.float32) / 9.0,
        np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32),
        np.array([[2, 0, -2],
                  [1, 0, -1],
                  [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0],
                  [-1, 0, 1],
                  [-2, 0, 2]], dtype=np.float32),
    ]


def show_original_and_feature_maps(original_gray: np.ndarray, feature_maps: list[np.ndarray], title: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(normalize_map(original_gray), cmap="gray")
    axes[0].set_title("Original / Input")
    axes[0].axis("off")

    for idx in range(4):
        axes[idx + 1].imshow(normalize_map(feature_maps[idx]), cmap="gray")
        axes[idx + 1].set_title(f"Feature Map {idx + 1}")
        axes[idx + 1].axis("off")

    plt.tight_layout()
    plt.show()


def show_final_output(original_gray: np.ndarray, final_map: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Final Output", fontsize=14)

    axes[0].imshow(normalize_map(original_gray), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(normalize_map(final_map), cmap="gray")
    axes[1].set_title("Final Combined Output")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def run_visual_pipeline(image_path: str | Path) -> None:
    image = np.array(Image.open(image_path))
    gray = rgb_to_gray(image)

    conv1_kernels = get_stage1_kernels()
    conv1_maps = [conv2d(gray, k, stride=1, padding=1) for k in conv1_kernels]
    show_original_and_feature_maps(gray, conv1_maps, "Stage 1: First Convolution")

    pool1_maps = [max_pool2d(normalize_map(m), size=2, stride=2) for m in conv1_maps]
    show_original_and_feature_maps(gray, pool1_maps, "Stage 2: First Max Pooling")

    conv2_kernels = get_stage2_kernels()
    conv2_maps = [conv2d(pool1_maps[i], conv2_kernels[i], stride=1, padding=1) for i in range(4)]
    show_original_and_feature_maps(gray, conv2_maps, "Stage 3: Second Convolution")

    pool2_maps = [max_pool2d(normalize_map(m), size=2, stride=2) for m in conv2_maps]
    show_original_and_feature_maps(gray, pool2_maps, "Stage 4: Second Max Pooling")

    final_output = np.mean(np.stack([normalize_map(m) for m in pool2_maps], axis=0), axis=0)
    show_final_output(gray, final_output)


def pick_image_and_run() -> None:
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not file_path:
        return

    try:
        run_visual_pipeline(file_path)
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to process image:\n{exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a simple CNN-style pipeline with 2 convolutions and 2 max-pooling steps."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to an image. If omitted, a file picker opens.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image:
        run_visual_pipeline(args.image)
    else:
        pick_image_and_run()


if __name__ == "__main__":
    main()
