# Predicting Antarctic Sea Ice with Scalable Deep Learning
## Ice-PatchNet
Ice-PatchNet is a novel, scalable deep learning framework designed to predict daily Sea Ice Extent (SIE) over the Antarctic region. By using a patch-based segmentation approach, the model can efficiently capture localized spatiotemporal features.

## Performance and Scope

- Target Region: Optimized for the Antarctic and Arctic polar regions.
- Resolution: Handles satellite imagery at a 25km spatial resolution.
- Predictive Windows: Supports lead-time predictions for 1-day, 7-day, and 14-day horizons.

### Important Version info
This version of Ice-PatchNet has only been tested on the images of sea ice extent over the Arctic region and the Antarctic region.

----------------------------------------------------------------------------

## Method Overview

1. **Image Preprocessing:** High-resolution SIE images are converted to grayscale to reduce complexity and computational overhead.
1. **Patch Segmentation:** The images are divided into a user-defined number of non-overlapping patches. This enables an equal distribution of SIE features and improves the model's ability to distinguish between land, open water, and ice.
2. **Spatiotemporal Feature Extraction:** Each patch is treated as a single unit and is processed through a deep convolutional neural network (3 layers, $3 \times 3$ kernels) with ReLU activation to identify localized patterns.
3. **Patches Reassembly:** Predicted patches are re-aligned using a global counter to reconstruct the full geographic extent without spatial distortion.

------------------------

## Citation
If you utilize this framework or the patch-based methodology in your research, please cite our ICDM '25 paper:
```
Amaraneni, S. V., Devnath, M. K., Srinivasa, S., Kulkarni, C., Chakraborty, S., & Janeja, V. P. (2025). Predicting Antarctic sea ice with scalable deep learning models. In Proceedings of the IEEE ICDM Workshop on Spatial and Spatiotemporal Data Mining (SSTDM).
```
