[Tutorial on Variational Autoencoders, Carl Doersch 2016](https://arxiv.org/pdf/1606.05908.pdf)

#### Problem
Estimate an unknown data distribution $$p(x)$$ with $$p(x) = \int p(x\mid z)q(z) dx$$. Now need to specify the form of $$z$$, ideally without hand-designs, and find out a tractable way to compute the integral. 
#### Objective
In VAE, $$q(z)$$ is a fixed prior distribution. $$p(x\mid z)$$ is sparse, and sampling-based methods like MCMC are slow to converge. Alternatively, VAE introduces an amortized posterior $$q(z\mid x)$$ to approximate $$p(z\mid x)$$ which enables fast sampling. 
According to [Eric Jang](https://blog.evjang.com/2016/08/variational-bayes.html), to measure how good the approximation is, "Mean-Field variational Bayes uses the Reverse KL Divergence as the distance metric between two distributions", which "measures the amount of information required to distort $$p(z\mid x)$$ to $$q(z\mid x)$$". 
Given $$x$$, the reverse KL divergence is
$$\mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log q(z\mid x) - \log p(z\mid x)\right]$$, which .
Bayesian rule says
$$\log p(z\mid x) = \log p(x\mid z) + \log p(z) - \log p(x)$$
Then $$\mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log q(z\mid x) -\log p(x\mid z) - \log p(z)\right] + \log p(x) = \mathcal{D}_\text{KL}(q(z\mid x)\|p(z)) -\mathbb{E}_q\left[\log p(x\mid z)\right] + \log p(x)$$.
After rearranging, 
$$\log p(x) - \mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log p(x\mid z)\right] - \mathcal{D}_\text{KL}(q(z\mid x)\|p(z))$$ (ELBO).
* Chosse $$q(z\mid x), p(z)$$ to be both multi-variant Gaussian, then the KL divergence can be analytically computed. 
* "It's important to keep in mind the implications of using reverse-KL when using the mean-field approximation in machine learning problems. If we are fitting a unimodal distribution to a multi-modal one, we'll end up with more false negatives (there is actually probability mass in P(Z) where we think there is none in Q(Z))."
#### The Information Theory Interpretation
$$\log p(x)$$ is the total number of bits required to construct $$x$$. The first term of ELBO is the number of bits neede to reconstruct $$x$$ from $$z$$, and the second term is the extra information needed to learn about $$x$$ when $$z$$ comes from $$q(z\mid x)$$ instead of $$p(z)$$, since $$p(z)$$ is a fixed prior containing no information about $$x$$. 
The first term "is a fairly waste-ful way to encode information about $$x$$: $$p(x\mid z)$$ does not model any of thecorrelations between the dimensions of $$x$$ under our model, so even an idealencoding must essentially encode every dimension separately."

