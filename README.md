# Depth-Consistent 3D Gaussian Splatting

**Depth-Consistent 3D Gaussian Splatting via Physical Defocus Modeling and Multi-View Geometric Supervision**

## Abstract

Three-dimensional reconstruction in scenes with extreme depth variations remains challenging. This work proposes a novel framework integrating depth-of-field supervision and multi-view consistency supervision to advance 3D Gaussian Splatting. Our method achieves superior depth fidelity through:
- Scale-recovered monocular depth estimation with defocus convolution
- LoFTR-based semi-dense feature matching for cross-view consistency
- Unified defocus physics with multi-view geometric constraints

**Key Achievement**: 0.8 dB PSNR improvement over state-of-the-art on Waymo Open Dataset.

## Pipeline

![Pipeline](images/pipeline/pipeline.png)

Our framework consists of two core components:
- **Depth-of-Field Supervision (Blue Flow)**: Addresses inaccuracies in distant scenes through physical defocus modeling
- **Multi-View Consistency Supervision (Orange Flow)**: Enforces cross-view geometric alignment via semi-dense matching

## Acknowledgments

We gratefully acknowledge the foundational work of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Kerbl et al., upon which this research builds.

## Publication

*To be published on arXiv*

```bibtex
@article{consistentgs2025,
  title={Depth-Consistent 3D Gaussian Splatting via Physical Defocus Modeling and Multi-View Geometric Supervision},
  author={[Authors]},
  journal={arXiv preprint},
  year={2025}
}
```
