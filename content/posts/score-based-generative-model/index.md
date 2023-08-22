---
title: "Score Based Generative Models"
summary: Introduce score based generative models
date: 2023-07-28
weight: 2
aliases: ["/score-based"]
tags: ["score-based"]
author: "Pu Zhang"
math: mathjax
---

### Intro

In the continually advancing world of image generation techniques, diffusion-centric methods have seen a notable rise in interest, alongside other significant progressions. Models such as [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release), [Imagen](https://imagen.research.google/), and DALL-E2(https://openai.com/dall-e-2) have garnered acclaim for their impressive feats. In this article, I'll demonstrate the connection between score-based generative models and these cutting-edge generative models. Our discussion will include a side-by-side review of diffusion models versus score-based generative models, illustrating the seamless integration of diffusion methods within the score-based paradigm. Moreover, I'll delve into captivating scenarios where classifiers can steer the diffusion mechanism, facilitating the creation of image examples influenced by specific class identifiers or textual cues. I utilized many of the explanations and illustrations found in [this post](https://yang-song.net/blog/2021/score/) written by the primary author of the score-based generative model paper.

### Generative Model History

In the domain of machine learning, models predominantly fall into two categories: generative and discriminative. A model that determines whether an image depicts a dog or a cat is discriminative, whereas a model that generates a lifelike image of a dog or cat is generative. Representative discriminative models encompass methods such as Logistic regression, SVM, random forest, and DNN. On the other hand, typical generative models consist of autoencoders, GANs, and diffusion models.

The underlying principle for all generative models is to transform a basic distribution, commonly a gaussian distribution, into a more intricate target distribution. This transformation simplifies the task of drawing samples from the intricate distribution to merely sampling from a gaussian distribution. As an illustration, the distribution from which our image dataset originates can serve as the target distribution. To put it more precisely, if our basic gaussian distribution is denoted as $X$, our objective is to devise a model such that $G(X)$ mirrors the distribution of our input dataset. Consequently, when we draw samples from $X$, $G(X)$ will reflect the distribution of our original data.


#### Autoencoder
Autoencoders represent an early popular generative model. They comprise two components: an encoder, which transforms input data into a latent space, and a decoder, which reverts this latent representation back to an output resembling the original input data. One can also interpret autoencoders as dimensionality reduction techniques, given that the latent space typically has significantly fewer dimensions than the input. This latent space acts as the architecture's bottleneck, ensuring only the most pertinent information is passed through and subsequently reconstructed. However, traditional autoencoders encode input data to arbitrary locations in the latent space, complicating the generation of realistic images upon decoding from sampled latent points. To address these limitations, the Variational Autoencoder (VAE) was introduced.

The VAE's innovation is its attempt to ensure encoded data within the latent space adheres to a standard normal distribution. This is accomplished by incorporating a KL-divergence between the encoded data and a standard normal distribution into the loss function. Thus, the loss consists of two components: a reconstruction loss (typically MSE, as with traditional autoencoders) and a KL-divergence loss. The resulting encoded data then aligns with a standard normal distribution. Such an approach guarantees that: 1) closely situated points in the latent space produce similar decoding outputs, and 2) any point sampled from the standard normal distribution yields meaningful decoded outputs. Nonetheless, the VAE often produces somewhat blurred results, likely due to inherent interpolations between various images.



![autoencoder-arch](images/autoencoder-arch.png)
*<center><font size="3">Autoencoder architecture(by Pu Zhang)</font></center>*


#### GAN
Generative Adversarial Networks (GANs) are a renowned class of generative models, characterized by the interplay between two main components: the generator and the discriminator. Introduced in 2014, the generator uses random noise to produce lifelike images, while the discriminator's role is to distinguish between fake images crafted by the generator and authentic ones. The loss function for the discriminator is the BCE loss, whereas the generator's loss aims to maximize the log likelihood across all fabricated samples. For optimal image generation with GANs, both the generator and the discriminator must be proficient by the end of training. Over the past decade, GANs have demonstrated the capability to create strikingly realistic images, marking a significant milestone in deep learning. However, they come with two primary challenges: 1) the complexity of training the model, and 2) potential mode collapse issues, causing the generated images to lack diversity, meaning the model may only replicate a narrow segment of the input data distribution, rather than its entirety.


![gan-arch](images/gan-arch.png)
*<center><font size="3">GAN architecture(by Pu Zhang)</font></center>*


#### Diffusion Model
The diffusion model, introduced in 2020, outperforms GAN in generating results. For an in-depth understanding of the diffusion model's functionality, please consult [my earlier article]({{< ref "/posts/diffusion-model/index" >}} "Diffusion Model").


#### Score Based Generative Model

In generative modeling, the goal is to mirror the intricate distribution of authentic data. Using a deep neural network (DNN) is a prevalent strategy to achieve this. The intention is to utilize the DNN to depict a complex probability distribution $P_\theta$. Nevertheless, the DNN's output $f_\theta$ doesn't directly reflect the distribution as it might not be positive throughout. Therefore, our initial action is to exponentiate the output. Subsequently, to shape a probability distribution ranging between 0 and 1, the output is normalized by dividing with a constant $Z_\theta$. This constant in the denominator is termed the normalizing constant. In Gaussian models, determining this constant is straightforward due to the simplistic nature of $f_\theta$. However, for advanced deep neural network architectures, it becomes a challenge to compute this constant.


![generative-model-normalization](images/generative-model-normalization.png)
*<center><font size="3">Model Normalization Challenge(by Pu Zhang)</font></center>*


The primary concept of the [score based generative modeling paper](https://arxiv.org/abs/2011.13456) revolves around score functions. This highlighted section of the illustration represents a density function alongside the score function for a combination of two Gaussian distributions. The density function is visually represented with varying shades, with a darker shade signifying a denser region. The score function, on the other hand, represents a vector field indicating the direction of the steepest increase in the density function. Given the density function, deducing the score function becomes straightforward by merely computing the derivative. In a similar vein, knowing the score function enables the retrieval of the density function, in essence, by calculating integrals. Hence, the score function and the probability distribution are interchangeable in their roles.

![score-density](images/score-density.png)
*<center><font size="3">Score Function vs Probability Density(by Pu Zhang)</font></center>*

Revisiting the challenge related to the normalizing constant, it becomes evident that when the gradient of the probability function is computed, the normalization constant turns to zero because it does not rely on the variable $x$ in question.
$$
\begin{aligned}
\nabla_x \log p_\theta(x) &= \nabla_x f_\theta(x) - \nabla_x \log Z_\theta \\\
&= \nabla_x f_\theta(x) \\\
&= s_\theta(x)
\end{aligned}
$$


#### Score Matching


Instead of employing a DNN to represent $p_\theta(x)$ outright, we utilize the DNN to characterize the score function $s_\theta(x)$. The goal now shifts to contrasting two vector fields associated with score functions. Subsequently, we can determine the disparity vectors for those vector pairs. Finally, by averaging over the densities of these disparity vectors, we derive a singular scalar-valued objective. This technique is recognized as the score matching algorithm.

![score-matching](images/score-matching.png)
*<center><font size="3">Score Matching Algorithm(by Pu Zhang)</font></center>*

In the score matching algorithm, we are given a set of samples from our input data distribution $$\{x_1, x_2, \dots, x_N\} \overset{\text{i.i.d.}}{\sim} p_{\text{data}}(x)$$. We want to use a score model $s_\theta(x)$ to estimate the score function of the input data distribution $$\nabla_x \log p_{\text{data}}(x)$$. To compare two vector fields of scores, our objective function is $$\frac{1}{2}E_{p_{\text{data}(x)}}\left[\\| \nabla_x \log p_{\text{data}}(x) - s_\theta(x) \\|\right]$$. This equation could be rewritten to not depend on $\nabla_{x}\log p_{data}(x)$ following integration by parts. To show how it works, let's use an 1D example for simplicity. 
$$
\begin{aligned}
-\int p(x) \nabla\_x \log p(x) s\_\theta(x) dx &= -\int p(x) \frac{\nabla\_x p(x)}{p(x)} s\_\theta(x) dx \\\
&= - \int \nabla\_x p(x) s\_\theta(x) dx \\\
&= -p(x) s\_\theta(x)\Big|\_{x=-\infty}^\infty + \int p(x) \nabla\_x s\_\theta(x) dx \\\
&= \int p(x) \nabla_x s_{\theta}(x) dx
\end{aligned}
$$
The last step for derivation holds because for most common probability density functions(pdfs), the value of the pdf approaches zero as $x$ approaches $\pm \infty$, so we have 
$$
\begin{aligned}
p(x) s\_\theta(x)\Big|\_{x=-\infty}^\infty &= p(\infty)s_\theta(\infty) - p(-\infty)s_\theta(-\infty) \\\
&= 0 \times s_\theta(\infty) - 0 \times s_\theta(-\infty) \\\
&= 0
\end{aligned}$$.
Plugin this into the objective of score matching, we have
$$
\begin{aligned}
\frac{1}{2}\left[ (\nabla_x \log p(x) - s_\theta(x))^2\right] &= \frac{1}{2}\int p(x)(\nabla_x \log p(x))^2 dx + \frac{1}{2}\int p(x)s_\theta(x)^2 dx \\\
                                                              &- \int p(x) \nabla_x \log p(x) s_\theta(x) dx \\\
                                                              &= {\color{Blue} \frac{1}{2}\int p(x)(\nabla_x \log p(x))^2 dx} + {\color{Orange} \frac{1}{2}\int p(x)s_\theta(x)^2 dx} \\\
                                                              &+ {\color{Green} \int p(x) \nabla_x s_{\theta}(x) dx}  \\\
                                                              &= {\color{Blue} \text{const}} + {\color{Orange} \frac{1}{2}E_{p(x)}[s_\theta(x)^2]} + {\color{Green} E_{p(x)} [\nabla_x s_{\theta}(x)]}
\end{aligned}
$$
The blue part above is constant because it does not depent on $\theta$. This illustration is in 1D case, in general, we have
$$
\small{
\begin{aligned}
\arg\min_{\theta} \frac{1}{2}E_{p_{\text{data}(x)}}[\\| \nabla_x \log p_{\text{data}}(x) - s_\theta(x) \\|] &= \arg\min_{\theta} E_{p_{data}(x)}\left[\frac{1}{2}\\| s_\theta(x) \\|^2 + \text{trace} ( \nabla_x s_\theta(x)) \right] \\\
                                                                                          &\approx \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^N \left[\frac{1}{2}\\| s_\theta(x) \\|^2 + \text{trace} ( \nabla_x s_\theta(x))\right]
\end{aligned}
}
$$


After determining the estimated score function, we must devise a method to construct our generative model by generating new data points from the given vector field of score functions. A potential strategy is to shift these points according to the directions suggested by the score function. Nevertheless, this won't yield valid samples as the points will ultimately converge. This challenge can be circumvented by adhering to a noisy rendition of the score function. In essence, we aim to introduce Gaussian noise into our score function and pursue those noise-distorted score functions. This technique is widely recognized as Langevin dynamics. More formally, have the following algorithm.

Our goal is to sample from $p(x)$ using only the score $\nabla_x \log p(x)$.
$$
\begin{aligned}
\text{Initialize} &\quad x^0 \sim \pi(x) \\\
\text{Repeat} &\quad t \leftarrow 1, 2, \dots, T \\\
&\quad z^t \sim \mathcal{N}(0, I) \\\
&\quad x_t \leftarrow x^{t-1} + \frac{\epsilon}{2} \nabla_x \log p(x^{t-1}) + \sqrt{\epsilon} z^t
\end{aligned}
$$
If $\epsilon \rightarrow 0$ and $T \rightarrow \infty$, we are guarantteed to have $x^T \sim p(x)$. The score function can be estimated via the score matching algorithm: $s_\theta(x) \approx \nabla_x \log p(x)$. 

*<center>![langevin](images/langevin.gif)</center>*
*<center><font size="3">Use Langevin dynamics for Sampling(by [Yang Song](https://yang-song.net/blog/2021/score/))</font></center>*

Directly applying score matching plus langevin dynamics does not give good results. This is because in low data density regions, the score function and the estiamted score function are not accurate, so it will be hard for Langevin Dynamics to navigate through those low-density regions.

*<center>![naive-smld](images/naive-smld.png)</center>*
*<center><font size="3">Challenges for Naive SMLD(by Pu Zhang)</font></center>*

To solve this challenge, we can inject Gaussian noise to perturb our data points. After adding enough Gaussian noise, we perturb the data points to everywhere in the space. This means the size of low data density regions becomes smaller. So in the context of image generation, adding additional Gaussian noise means we inject Gaussian noise to perturb each pixel of the image. For this toy example in the picture below, you can see that, after injecting the right amount of Gaussian noise, the estimated scores become accurate almost everywhere.
But simply injecting Gaussian noise will not solve all the problems. Because of perturbation of data points, those noisy data distances are no longer good approximations to the original true data density.


*<center>![pertubed-smld](images/pertubed-smld.png)</center>*
*<center><font size="3">SMLD on Pertubed Data(by Pu Zhang)</font></center>*

To solve this problem, we can use a multiple sequence of different noise levels. Here we use Gaussian noise with mean 0 and standard deviation from $\sigma_1$ to $\sigma_3$ to perturb our training data set($\sigma_1 < \sigma_2 < \sigma_3$). And this will give us three noisy training data sets. For Each noisy data set, there will be a corresponding noise data density, which we denote as $p_{\sigma_1}(x)$ to $p_{\sigma_3}(x)$. 

*<center>![multi-level-pertubed-distribution](images/multi-level-pertubed-distribution.png)</center>*
*<center><font size="3">Multi-level Noise Pertubed Data Distribution(by Pu Zhang)</font></center>*


To estimate the score function, instead of training separate score models on different noise levels, we use a noise conditional score model, which takes noise level $\sigma$ as one additional input dimension to the model. We use score matching to jointly train the score model across all levels. If our optimizer is powerful enough, and if our model is expressive enough, we will obtain accurate score estimation for all noise labels.
During sampling time, we can first apply Langevin dynamics to sample from the score model with the biggest perturbation noise. And the samples will be used as the initialization to sample from the score model of the next noise level. And then we continue in this fashion until finally we generate samples from the score function with the smallest noise level.


*<center>![multi-level-pertubed-loss](images/multi-level-pertubed-loss.png)</center>*
*<center><font size="3">Multi-level Noise Pertubed Model and Loss(by Pu Zhang)</font></center>*

---