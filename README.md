# EDGNN

## Overview

Our work investigates how directionality in brain functional networks influences graph neural network performance for fMRI-based mental disorder diagnosis.
We propose a directed graph learning framework that integrates Granger causality estimation with graph convolution, enabling more accurate modeling of brain connectivity.
At the same time, we further introduce an efficient model that maintains the advantages of direction-aware graph representations while improving computational efficiency.

This repo now provides a demo implementation of our proposed method EDGNN for directed graph learning based on Granger causality.
Detailed experimental results are presented in the supplementary material.

<p align="center">
  <img src="data/architect.jpg" alt="Model architecture" width="60%">
</p>

Due to permission reasons, we cannot open source the patient's original BOLD signal. This demo uses the ABIDE dataset pre-processed by the JNGC method, which contains 1090 samples for training.
