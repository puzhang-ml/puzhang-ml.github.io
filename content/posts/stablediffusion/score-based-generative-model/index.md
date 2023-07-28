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

next one:
$$
\begin{equation}
  x = a_0 + \cfrac{1}{a_1 
          + \cfrac{1}{a_2 
          + \cfrac{1}{a_3 + \cfrac{1}{a_4} } } }
\end{equation}
$$

yet another one:
$$
A_{m,n} = 
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
 \end{pmatrix}
$$

$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)
$$


---