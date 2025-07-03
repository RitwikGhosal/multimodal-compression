# Multimodal Compression via Shared Latent Representations

This project implements a multimodal autoencoder that learns a **shared latent space** for images and their corresponding text captions. 
The goal is to jointly compress both modalities using a common representation and reconstruct them from that shared latent vector.

Built using PyTorch and pretrained models (ResNet18, DistilBERT), the architecture learns compact latent codes that capture both visual and textual semantics.

---
## Components

- `Flickr8kDataset`: Loads image-caption pairs, tokenizes text using HuggingFace tokenizer.
- `ImageEncoder`: Pretrained ResNet18, frozen.
- `TextEncoder`: Pretrained DistilBERT, frozen.
- `FusionModule`: MLP that merges image and text features into a shared 256-dim latent `z_shared`.
- `CaptionDecoder`: GRU-based decoder for generating captions from `z_shared`.
- `ImageDecoder`: Transposed-convolutional decoder to reconstruct the original image.
- `training on`: Reconstruction loss (MSE) and caption loss (CrossEntropy).

---
## Dataset

- **Flickr8k** dataset
  - Images: `.jpg` format
  - Captions: `.txt` file (image_name \t caption)
