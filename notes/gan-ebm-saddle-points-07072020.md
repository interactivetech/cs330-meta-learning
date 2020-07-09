Adapted from John Schulman's notes.

#### Energy-Based Models
$$p(x) = \exp(-c(x))/Z_c$$
Negative log likelihood per datapoint:
$$\mathcal{L} = \mathbb{E}_{p_\text{data}}\left[c(x)\right] + \log(Z_c)$$
With importance sampling and Jensen's inequality, 
$$\log(Z_c) = \log \int dx \exp(-c(x)) = \log \mathbb{E}_q\left[\exp(-c(x)) / q(x)\right] \geq \mathbb{E_q}\left[-c(x)- \log q(x)\right]$$
Combining above, 
$$\mathcal{L} \geq \mathbb{E}_{p_\text{data}}\left[c(x)\right] - \mathbb{E}_q\left[c(x)\right] + \mathcal{H}(q)$$
with the equality held when $$q(x) = \exp(-c(x))/Z_c$$.
Thus rewrite the objective of minimizing negative loglikelihood as
$$\min_c \max_q \mathcal{L}(c, q) = \mathbb{E}_{p_\text{data}}\left[c(x)\right] - \mathbb{E}_q\left[c(x)\right] + \mathcal{H}(q)$$. 
* The optimization could be unstable with SGD because $$q$$ may not catch up with $$c$$. 
* Introduce a prior on $$c$$, which is equivalent to a regularization term $$\phi(c)$$ that encourages $$c$$ to be small.

#### Deriving GAN form EBM
Let $$c(x) = \log\sigma(-f(x)), \phi(c) = \mathbb{E}_{p_\text{data}}\left[-\log\sigma(f(x)) - \log\sigma(-f(x))\right]$$, where $$\sigma(\cdot)$$ is the sigmoid function. 
Then $$\mathcal{L}(f, q) = -\mathbb{E}_{p_\text{data}}\left[\log\sigma(f(x))\right] -\mathbb{E}_q\left[\log(1-\sigma(f(x)))\right] + \mathcal{H}(q)$$
* Compared to the original GAN formulation, with min-max rather than max-min, given $$f$$, one cannot recover $$q$$ if there is no entropy regularization term. Although they both yield the same optimal value. 
* With the choice of $$c(x)$$, $$Z_c$$ could be infinite. This can be fixed by having $$\mathcal{D}_\text{KL}(q, q_0)$$ where $$q_0$$ is Gaussian instead of $$\mathcal{H}(q)$$. 
* $$\min_q \mathcal{L}(c, q) = \mathcal{D}_\text{JS}(p_\text{data}, q)$$. The result can be generalized to $$f$$-divergence with $$c(x) = \phi(-h(x))$$.