# PPRL-VGAN
Implementation of ["VGAN-Based Image Representation Learning for Privacy-Preserving Facial Expression Recognition"](https://arxiv.org/pdf/1803.07100.pdf).

# Prerequisites
1) Tensorflow 1.0.1 or above
2) Keras 2.0.2
3) [FERG facial expression database](https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html)
4) [MUG facial expression database](https://mug.ee.auth.gr/fed/)

# Usage
First download both datasets from the offical sources. Then, reduce the video resolution to 64x64x3, and save the data into ".mat" or ".hdf5" files.
In our experiments, we used 85% of the frames in each video for training and the rest frames for testing. You can decide your own training/testing ratio.

**Training**
```python
python train.py
```
**Evaluation of the identification and expression recognition performance**
```python
python Privacy_vs_Utility_check.py
```

**Image processing applications(e.g., expression morphing)**

```python
python img_applications.py
```

# Project webpage

coming soon...
