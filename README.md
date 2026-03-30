
# CNN Visualizer Project

This project demonstrates a step-by-step visualization of how a simple convolutional neural network processes an image. Starting from the original input, the image is passed through a first convolution stage where different filters extract low-level features such as edges and textures, followed by max pooling which reduces the spatial size while preserving the strongest responses. A second convolution is then applied to these feature maps, combining earlier patterns into more abstract representations, and another max pooling step further compresses the information by keeping only dominant structures. Finally, the resulting feature maps are combined into a single output that represents a simplified and abstract version of the original image, illustrating how CNNs progressively transform raw pixel data into compact, meaningful features.


It takes one image and shows the processing steps of a small CNN-style pipeline:

1. Original image
2. First convolution (4 feature maps)
3. First max pooling (4 feature maps)
4. Second convolution (4 feature maps)
5. Second max pooling (4 feature maps)
6. Final combined output

## Important note

This is a visualization project, not a trained neural network.
It uses fixed kernels so you can clearly see what convolution and max pooling do.

## Files

- `cnn_visualizer.py` — main script
- `requirements.txt` — dependencies

## Install

```bash
pip install -r requirements.txt
```

## Run

### Option 1: choose image with file picker

```bash
python cnn_visualizer.py
```

### Option 2: pass an image directly

```bash
python cnn_visualizer.py --image path/to/your/image.jpg
```
