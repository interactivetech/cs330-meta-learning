[Variaitonal Inference: Foundations and Innovations, David Blei 2017](https://www.youtube.com/watch?v=Dv86zdWjJKQ)
#### Problem
A probabilistic model: $$p(z, x)$$ where $$z$$ is the latent variable, $$x$$ observed variable. 
Inference: $$p(z\mid x) = p(z, x) / p(x)$$, where $$p(x)$$ is usually untractable. Thus need approximated posterior inference. 

#### Variational Inference (VI)
Posit a variational family of distributions: $$q(z; \nu)$$, minimize $$\mathcal{D}_{KL}(q(z; \nu)\|p(z\mid x))$$. 
* VI turns inference into optimization. 
* Divergence other than KL leads to EP, BP, etc. KL makes the optimization possible but has its problems. 

#### Conditinally Conjugate Models
Setup: global variable $$\beta$$, local variables $$z_i, x_i$$, $$p(\beta, z, x) = p(\beta)\prod_{i=1}^n p(z_i, x_i\mid \beta)$$. 
Assume $$p(z_i\mid \beta, x_i), p(\beta \mid z, x)$$ (called complete conditionals) are in the exponential family. Explicitly, 
$$p(z_i\mid \beta, x_i) = h(z_i) \exp\{n_l(\beta, x_i)^T z_i - \alpha (\eta_l(\beta, x_i))\}$$,
$$p(\beta\mid z, x) = h(\beta)\exp\{\eta_g(z, x)^T \beta - \alpha (\eta_g(z, x))\}$$. 
The global parameter comes form conjugacy: 
$$\eta_g(z, x) = \alpha + \sum_{i=1}^n t(z_i, x_i)$$, where $$t(\cdot)$$ is the sufficient statistics, $$\eta_g$$ parametrizes the global paramter e.g. the bias of a coin, which can be inferred by observing statistics of tosses. 
By introducing parameter $$\nu$$, the objective becomes $$\min \mathcal{D}_{KL}(q(\beta, z; \nu)\|p(\beta, z\mid x))$$.
* KL is intractable. Minimizing KL is equivalent to maximizing ELBO: 
$$\mathcal{L}(\nu) = \mathbb{E}_q\left[\log p(\beta, z, x)\right] - \mathbb{E}_q \left[\log q(\beta, z; \nu)\right]$$. 
* ELBO = maximum likelihood + entropy. 
* ELBO is a lower bound on evidence likelihood $$\log p(x)$$. 

#### Mean-Field VI
Now need to specify the form of $$q(\beta, z; \nu)$$. Use the Mean-Field family: $$q(\beta, z; \lambda, \phi) = q(\beta; \lambda) \prod_{i=1}^n q(z_i; \phi_i)$$. 
* The mean-field failiy is fully factorized. 
* Assert that each parameter factor is in the same familiy as the model's comlete conditionals. For example, $$p(\beta\mid z, x), q(\beta; \lambda)$$ are both in the exponential familiy. 
* Doing coordinate-ascent VI amounts to 
$$\lambda^* = \mathbb{E}_\phi\left[\eta_g(z, x)\right]; \phi_i^* = \mathbb{E}_\lambda \left[\eta_l(\beta, x_i)\right]$$. 
* Iteratively update each parameter, holding others fixed. 
* ELBO is not convex, so this process will find a local optima. 
* $$\eta_g(z, x)$$ is computed using the conjugacy result. 
* [Q] How is $$\eta_l$$ computed? Why there is expetation over $$\lambda$$?

#### Stochastic VI
* Classical VI: based on Mean-Field VI, do local computation for each datapoint, then re-estimate the global structure. This is inefficient. 
* Scale up by submsampling data. 
* SGD: replace the gradient with cheaper noisy estimates. 
    * Requires unbiased gradients and step size. 
* Stochastic VI uses a noisy version of natural gradient of ELBO

#### Black-Box VI
* Non-Conjugate models: $$p(\beta, z, x) = p(\beta)\prod_{i=1}^n p(z_i, x_i\mid \beta)$$ with no other assumptions (e.g. exponenetial family as in Conjugate Models).
* The objective can't be computed analytically and thus the gradient. To evaluate gradient, use REINFORCE gradient (score gradient): $$\nabla_\nu \mathcal{L} = \mathbb{E}_{q(z; \nu)}\left[\nabla_\nu \log q(z; \nu) (\log p(x, z) - \log q(z; \nu)\right]$$

#### Reparameterization
$$z\sim q(z; \nu)$$ is reparameterized as $$\epsilon \sim s(\epsilon), z = t(\epsilon, \nu)$$, e.g. the Gaussian reparameterization trick in VAE. 
Then the gradient of ELBO becomes $$\nabla \mathcal{L} = \mathbb{E}_{s(\epsilon)}\left[\nabla_z\left[\log p(s, z) - log q(z; \nu)\right]\nabla_\nu t(\epsilon, \nu)\right]$$, so that the expectation does not depend on $$\nu$$ and MC is efficient. 

#### Amortization
$$\nu = \nu(x)$$ as in VAE. Fast inference without running optimization.

[//]: #
   [dill]: <https://github.com/joemccann/dillinger>
