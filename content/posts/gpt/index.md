---
title: "GPT"
summary: Introduce how GPT works
date: 2023-08-22
weight: 2
aliases: ["/gpt"]
tags: ["gpt"]
author: "Pu Zhang"
math: mathjax
---


### Intro

Recently, ChatGPT has gained siginificant attention, with some drawing parallels to the invention of the internet. In this article, I will dive deep into the principles of ChatGPT, from its first to third iterations, using the three famous papers about ChatGPT as references:
- ChatGPT-1: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- ChatGPT-2: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- ChatGPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

All ChatGPT versions share the similar architecture as transformers. For details on transformers, you can refer to [my earlier article]({{< ref "/posts/transformers/index" >}} "Transformers").

### ChatGPT-1

Let's first talk about the ChatGPT-1 architecture. This serves as the foundation to subsequent iterations of ChatGPT.

---
