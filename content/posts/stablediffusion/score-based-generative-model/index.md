---
title: "Score Based Generative Models"
summary: Introduce score based generative models
date: 2023-07-27
weight: 2
aliases: ["/score-based"]
tags: ["score-based"]
author: "Pu Zhang"
math: mathjax
---

### Intro

- **We'll be using `yml/yaml` format for all examples down below, I recommend using `yml` over `toml` as it is easier to read.**

- You can find any [YML to TOML](https://www.google.com/search?q=yml+to+toml) converters if necessary.

### Home-Info Mode

![homeinfo](images/homeinfo.jpg)

Use 1st entry as some Information

add following to config file


$$
  \begin{bmatrix}
    a & b \\
    c & d \\
    e & f \\
  \end{bmatrix}
$$

$$
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

---