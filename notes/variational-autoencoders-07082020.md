[Tutorial on Variational Autoencoders, Carl Doersch 2016](https://arxiv.org/pdf/1606.05908.pdf)

#### Problem
Estimate an unknown data distribution $$p(x)$$ with $$p(x) = \int p(x\mid z)q(z) dx$$. Now need to specify the form of $$z$$ and find out a tractable way to compute the integral. 
#### Objective
In VAE, $$q(z)$$ is a fixed prior distribution. Since $$p(x\mid z)$$ is sparse, VAE introduces an amortized posterior $$q(z\mid x)$$ to speed up sampling. 
Given $$x$$, 
$$\mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log q(z\mid x) - \log p(z\mid x)\right]$$
$$\log p(z\mid x) = \log p(x\mid z) + \log p(z) - \log p(x)$$
Then $$\mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log q(z\mid x) -\log p(x\mid z) - \log p(z)\right] + \log p(x) = \mathcal{D}_\text{KL}(q(z\mid x)\|p(z)) -\mathbb{E}_q\left[\log p(x\mid z)\right] + \log p(x)$$.
Rearranging, 
$$\log p(x) - \mathcal{D}_\text{KL}(q(z\mid x)\|p(z\mid x) = \mathbb{E}_q\left[\log p(x\mid z)\right] - \mathcal{D}_\text{KL}(q(z\mid x)\|p(z))$$ (ELBO).
Chosse $$q(z\mid x), p(z)$$ to be both multi-variant Gaussian, then the KL divergence can be analytically computed. 
#### The Information Theory Interpretation
$$\log p(x)$$ is the total number of bits required to construct $$x$$. The first term of ELBO is the number of bits neede to reconstruct $$x$$ from $$z$$, and the second term is the extra information needed to learn about $$x$$ when $$z$$ comes from $$q(z\mid x)$$ instead of $$p(z)$$, since $$p(z)$$ is a fixed prior containing no information about $$x$$. 
The first term "is a fairly waste-ful way to encode information about $$x$$: $$p(x\mid z)$$ does not model any of thecorrelations between the dimensions of $$x$$ under our model, so even an idealencoding must essentially encode every dimension separately."
