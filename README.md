# SB-GRPO: Safety-Balanced GRPO

This repository contains the final project for **CS 285: Deep Reinforcement Learning (UC Berkeley, Spring 2026)**.

## Overview

Aligned language models must balance refusing harmful requests while remaining helpful on benign ones. However, modern models often exhibit over-refusal, unnecessarily rejecting harmless prompts. We propose **Safety-Balanced GRPO (SB-GRPO)**, a method that incorporates representation-level signals into reinforcement learning.

We first identify directions in the model’s hidden space corresponding to:
1. true refusal for harmful prompts  
2. false refusal for benign prompts  

These directions are computed using activation statistics from a frozen model. We then introduce a **geometry-based reward** that:
- encourages alignment with the true refusal direction for harmful prompts  
- discourages alignment with the false refusal direction for benign prompts  

Training is performed using **Group Relative Policy Optimization (GRPO)**.

## Abstract

> Aligned language models must balance refusing harmful requests while remaining helpful on benign ones. However, modern models often exhibit over-refusal, unnecessarily rejecting harmless prompts. We propose Safety-Balanced GRPO (SB-GRPO), a method that incorporates representation-level signals into reinforcement learning. We first identify directions in the model’s hidden space corresponding to (1) true refusal for harmful prompts and (2) false refusal for benign prompts, using activation statistics from a frozen model. We then introduce a geometry-based reward that encourages alignment with the true refusal direction for harmful prompts and discourages alignment with the false refusal direction for benign prompts during training with Group Relative Policy Optimization (GRPO). Our implementation includes pseudo-aware layer selection, orthogonalization of false-refusal directions, label-gated geometry rewards, and completion-level hidden state pooling. Experiments on harmful and over-refusal benchmarks show that label-gated geometry rewards substantially reduce over-refusal while maintaining strong harmful-request refusal. Our method provides a principled way to control refusal behavior through internal representations rather than surface-level heuristics alone.
