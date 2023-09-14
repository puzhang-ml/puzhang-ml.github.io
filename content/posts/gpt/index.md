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

Recently, ChatGPT has gained siginificant attention, with some drawing parallels to the invention of the internet. In this article, I will dive deep into the principles of GPT, from its first to third iterations, using the three famous papers about GPT as references:
- GPT-1: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT-2: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

All these GPT versions share the similar architecture as transformers. For details on transformers, you can refer to [my earlier article]({{< ref "/posts/transformers/index" >}} "Transformers").

### GPT-1

Let's first talk about the GPT-1. This serves as the foundation to subsequent iterations of GPT. The GPT acronym actually comes from generative pre-training and generative pre-training is basically a self-supervised learning. If you take a book which has let's say roughly 10,000 words you can actually chop it into the window of let's say 100, you can predict the next next word and moving this window you can actually create a lot of data set for your model. In this method you don't actually need to manually label this data because it's already labeled itself, in other words, the sequence structure. [In the paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), we pre-train the model using 7,000 books, then the model learns the underlying structure of the language and can create the sentences using the structure of the language. Then we fine tune the model for different taks where we have labels. In this second stage, instead of predicting the next word, we predict the label of the task. 

[In the paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), the fine tuned models actually beat the state-of-the-art models that are explicitly trained on different tasks on nine of the data sets. Most of the deep learning methods require substantial amounts of manually labeled data which restricts the application in many domains that suffer from scarsity of annotated resources. So pre-training and then fine-tuning can solve this issue.

---
