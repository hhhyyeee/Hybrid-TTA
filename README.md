# Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection [ICCV 2025]

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://sites.google.com/view/hybrid-tta/home)  [![arXiv](https://img.shields.io/badge/arXiv-2409.08566-b31b1b.svg)](https://arxiv.org/abs/2409.08566)

[Hyewon Park](https://github.com/hhhyyeee)<sup>1</sup> [Hyejin Park](https://github.com/kunsaram01)<sup>1</sup> [Jueun Ko](https://github.com/0ju-un)<sup>1</sup> Dongbo Min<sup>1</sup>

<sup>1</sup> Ewha W. University

<!-- :scroll: Source code for [**Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection**](https://arxiv.org/abs/2409.08566), **ICCV 2025**. -->

🚨 **Source code of Hybrid-TTA will be updated soon.** 

## Abstract

Continual Test Time Adaptation (CTTA) has emerged as a critical approach to bridge the domain gap between controlled training environments and real-world scenarios. Since it is important to balance the trade-off between adaptation and stabilization, many studies have tried to accomplish it by either introducing a regulation to fully trainable models or updating a limited portion of the models. This paper proposes \textbf{Hybrid-TTA}, a holistic approach that dynamically selects the instance-wise tuning method for optimal adaptation. Our approach introduces Dynamic Domain Shift Detection (DDSD), which identifies domain shifts by leveraging temporal correlations in input sequences, and dynamically switches between Full or Efficient Tuning for effective adaptation toward varying domain shifts. To maintain model stability, Masked Image Modeling Adaptation (MIMA) leverages auxiliary reconstruction task for enhanced generalization and robustness with minimal computational overhead. Hybrid-TTA achieves $0.6\%p$ gain on the Cityscapes-to-ACDC benchmark dataset for semantic segmentation, surpassing previous state-of-the-art methods. It also delivers about 20-fold increase in FPS compared to the recently proposed fastest methods, offering a robust solution for real-world continual adaptation challenges.

