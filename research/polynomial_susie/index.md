---
title: "Polynomial approximation SuSiE"
description: "We extend the sum of single effects regression to support arbitrary likelihood and link function by representing (perhaps approximately) the log likelihood of each observations as a polynomial in the linear prediction. Once this approximation is made we can treat inference in multiple likelihoods with a uniform inference procedure. Furthermore, we can make the polynomial approximation arbitrarily precise by increasing the degree of the polynomial approximation."

author: "Karl Tayeb"
date: "3/15/23"
format: 
  html: 
    code-fold: true
    keep-md: true
execute:
  cache: true
---



## Introduction

The sum of single effects (SuSiE) regression, is a regression with a SuSiE prior on the effects. For a Gaussian model with identity link, using a variational approximation that factorizes over single effects yields a fast coordinate ascent variational inference scheme which can be optimized by fitting a sequence of single effect regression (SERs). The computational simplicity of this result relies on (1) the variational approximation and (2) the fact the the log likelihood is quadratic in the regression coefficients.

Here, we consider the more general case where the log-likelihood is a polynomial in the regression coefficients. We can approximate the log-likelihood of models with many choices of likelihood and link function with polynomials, so having a fast inference scheme for this case provides a uniform treatment of extensions of SuSiE to different observation models. We use the same variational approximation as in linear SuSiE. Approximating the log-likelihood as as polynomial is sufficient to

## Preliminaries
### Polynomial shift

Given the coefficients ${\bf a}$ we would like to find the coefficients for the change of variable

$$\phi_{\bf a}(x + y) = \phi_{{\bf b}({\bf a}, y)}(x)$$

$$
\begin{aligned}
\phi_{\bf a}(x + y) 
&= \sum_{m=0}^M a_m(x + y)^m \\
&= \sum_{m=0}^M a_m \sum_{k=0}^m {m \choose k} x^k y^{m-k} \\
&= \sum_{k=0}^M \left(\sum_{m=k}^M a_m {m \choose k}y^{m-k} \right) x^k \\
&= \sum_{k=0}^M {\bf b}({\bf a}, y)_k x^k, \quad  {\bf b}({\bf a}, y)_k := \left(\sum_{m=k}^M a_m {m \choose k}y^{m-k} \right)
\end{aligned}
$$
Inspecting the expression for ${\bf b}({\bf a}, y)_k$ we can see that the new coefficient vector ${\bf b}({\bf a}, y)$ can be written compactly in matrix form

$$
{\bf b}({\bf a}, y) = M(y) {\bf a}, \quad M_{ij} = {j \choose i} y^{j -i}\;  \forall j \geq i, 0 \text{ otherwise}
$$
Supposing $y$ is random, we will also have need to compute the expected value of ${\bf b}({\bf a}, y)$.
For $y \sim p$, $\mathcal M(p) = \mathbb E_p[M(y)]$, we can compute these expected coefficients

$$
{\bf c}({\bf a}, p) = \mathcal M(p) {\bf a}
$$




### Polynomial rescaling

$$\phi_{\bf a}(cx) = \phi_{{\bf d}({\bf a}, c)}(x)$$

$$
\sum_m a_m(c x)^m = \sum_m (a_m c^m) x^m = \sum_m {\bf d}({\bf a}, c)_m x^m
$$


## Model

$$
\begin{aligned}
y_i | \mu_i &\sim f(\mu_i, \theta) \\
\mu_i &= g({\bf x}^T_i \beta ) \\
\beta &\sim \text{SuSiE}(L, \{\sigma_{0l}\})
\end{aligned}
$$
$f$ is the observation model, parameterized by $\mu$ and $\theta$. 
$g$ is a link function which maps the linear predictions $\psi_i := {\bf x}_i^T \beta$ to the parameter $\mu_i$.


### Polynomial approximation to the log-likelihood

For each observation $y$, we can approximate the log-likelihood as a polynomial in the linear prediction. $f$ is a polynomial of degree $M$

$$
\log p(y | \psi) = \log f(y | g(\psi)) \approx f(\psi)
$$

$$
f(\psi) = \sum_{m=0}^M a_m \psi^m
$$

There are many ways to construct this approximation. At a high level, we want the approximation to be "good" (some measure of error between the approximate log-likelihood and the exact log-likelihood is small) at plausible values of $\psi$. So far, if have used a truncated Chebyshev series.

We write $f_i(\psi)$ as the polynomial approximation for $\log p(y_i | \psi)$. We write it's coefficients $\left( a_m^{(i)} \right)_{m \in [M]}$. To denote a polynomial with coefficients ${\bf a}$ we will write $\phi_{\bf a}$, e.g. $f(\psi) = \phi_{\bf a}(\psi)$.

We have a polynomial approximation for each observations. Let ${\bf a}_i$ denote the coefficients for observations $i$. and $A$ be the $n \times m$ matrix where each row corresponds to an observation.




## Single effect regression with polynomial approximation

### SER prior

The SER prior $\beta \sim SER(\sigma_0^2, \pi)$ 

$$
\begin{aligned}
\beta = b\gamma\\
b \sim N(0, \sigma_0^2) \\
\gamma \sim \text{Mult}(1, \pi)
\end{aligned}
$$

### Polynomial approximation for SER posterior

The SER follows from $p$ univariate regressions

$$
\begin{aligned}
p(b | {\bf y}, X, \gamma=j, \sigma^2_0) 
  &= p(b | {\bf x}_j, {\bf y}, \sigma^2_0) \\
  &\propto p({\bf y}, b| {\bf x_j}, \sigma^2_0) \\
  &\approx \exp\{ \sum_i \phi_{\bf a_i}(x_{ij} b) + \log p(b)\}
\end{aligned}
$$

Supposing $\log p(b) \approxeq \phi_{\bf \rho}(b)$ for some coefficients $\rho$, the unnormalized log posterior density can be written as a degree $M$ polynomial with coefficients ${\bf f}(A, {\bf x}_j, \rho)$ defined below:

$$
\begin{aligned}
\sum_i \phi_{\bf a_i}(x_{ij} b) + \phi_{\bf \rho}(b) &= \sum_i \phi_{{\bf d}({\bf a}_i, x_{ij})}(b) + \phi_{\bf \rho}(b), \\
&= \phi_{{\bf f}(A, {\bf x}_j, \rho)}(b), \quad {\bf f}(A, {\bf x}_j, \rho) := \sum_i {\bf d}({\bf a}_i, x_{ij}) + \rho. \\
\end{aligned}
$$

For clarity we write ${\bf f}_j$ for ${\bf f}(A, {\bf x}_j, \rho)$. The posterior density is $q(b) = \frac{1}{Z_j}\exp\{\phi_{{\bf f}_j}(b)\}$ where $Z_j = \int_{\mathbb R} \exp\{\phi_{{\bf f}_j}(b)\} db$. The normalizing constant may be computed numerically. Or, we can take a Gaussian approximation for $q(b)$.


$$
\begin{aligned}
p(\gamma = j | {\bf y}, X, \sigma^2_0) 
&\propto p({\bf y} | {\bf x}_j, \gamma = j) \pi_j \\
&= \left(\int p({\bf y}, b | {\bf x}_j, \gamma = j) db\right) \pi_j \\
&= \left(\int \exp\{\sum_i \phi_{{\bf a}_i} (x_{ij} b) + \phi_{\rho}(b) \}\right) \pi_j \\
&= \left(\int \exp\{\phi_{{\bf f}_j}(b) \} db\right) \pi_j = Z_j \pi_j \\
\end{aligned}
$$

$$\alpha_j := q(\gamma = j) = \frac{Z_j \pi_j}{\sum_k Z_k \pi_k}$$

### Variational objective

It will be useful to write out the variational objective for the SER:

$$
F_{SER}(q| {\bf y}, X, \sigma^2_0) = \mathbb E_q[\log p({\bf y} | X, b, \gamma, \sigma^2_0)] - KL[q || p].
$$
The exact posterior maximizes $F_{SER}$, that is 

$$
p_{post} = \arg\max_q F_{SER}(q| {\bf y}, X, \sigma^2_0).
$$

We can approximate the variational objective by substituting our polynomial approximation

$$
\hat F_{SER}(q| A, X, \sigma^2_0) = \mathbb E_q \left[
\sum_i \phi_{\exp\{{\bf a}_i}(X (b\gamma)) + \log p(b | \sigma^2_0) + \log p(\gamma | \pi)\}
\right]  - KL[q || p].
$$

As discussed above, $q(b | \gamma) \propto \exp\{\phi_{{\bf f}_j}(b)\}$ and $q(\gamma = j) = \frac{Z_j}{\sum_k Z_k}$.

Let $SER(A, X, \sigma^2_0, \pi)$ be a function that returns $q$. 

### Polynomial to Gaussian

For the extention to SuSiE, we are going to need to compute $\mu_j^k = \mathbb E[b^k | \gamma = j] = \int_{\mathbb R} b^k \exp\{\phi_{{\bf f}(A, {\bf x}_j, \rho)}(b)\} db$. We can ensure that this function is integrable by making selecting polynomial approximations ${\bf a}_i$ which are of even degree and $a_{iM} < 0$. This ensures that the leading coefficient of ${\bf f}$ is negative and that $e^{\bf f}$ decays. However, this integral may not be available analytically in general.

However, if $\phi_{\bf f}$ is degree $2$, this is a special case where the moments of a Gaussian distribution can be computed analytically. This motivates us to "approximate the approximation".

We take a Laplace approximation to $\phi_{\bf f}$. Currently I approximate the degree $M$ polynomial $\phi_{\bf f}$ with the degree $2$ polynomial by the following: first we search for the mode of $\phi_{\bf f}$ e.g. by finding the roots of $\phi_{\bf f}^\prime$ and selecting the one that maximizes $\phi_{\bf f}$ (although there may be cheaper ways to do this?). Then, we take a second order Taylor series expansion around the mode.

It's worth noting that these two levels of approximation result in a different strategy than taking a "global" second order approximation directly, or a Laplace approximation $\log p(y, \beta) \rightarrow {\bf f} \rightarrow {\bf g}$. "Laplace approximation of a degree-$M$ polynomial approximation" is a local approximation around the approximate mode. Directly applying Laplace approximation would result in a local approximation around the exact mode. We would prefer the latter except that (1) the polynomial approximation provides the computational simplifications we need to extend to SuSiE and (2) finding the posterior mode and it's second derivative may be more expensive in the general case.


## Polynomial SuSiE
### Variational approximation

We will use the variational approximation that factorizes over single effects. This is the same variational approximation used in linear SuSiE

$\beta_l = \gamma_l b_l$
$$q(\beta_1, \dots, \beta_L) = \prod_l q(\beta_l)$$

### Variational objective

We want to find $q$ that optimizes the **evidence lower bound** (ELBO) $F$

$$
F_{\text{SuSiE}}(q| {\bf y}, X, \{\sigma^2_{0l}\}, \{\pi_l\}) = \mathbb E_q[\sum \log p(y_i | {\bf x}_i, \beta)] - \sum KL[q(\beta_l) || p(\beta_l | \sigma^2_{0l}, \pi_l)]
$$

Evaluating the expectations $\mathbb E_q[\log p(y_i | {\bf x}_i, \beta)]$ is generally hard, we make progress by substituting the polynomial approximation, we denote the approximate ELBO $\hat F$.

$$
\begin{aligned}
\hat F_{\text{SuSiE}}(q| A, X, \{\sigma^2_{0l}\}, \{\pi_l\})
&= \mathbb E_q[\sum_i \phi_{{\bf a}_i}({\bf x}_i^T \beta)]
- \sum KL[q(\beta_l) || p(\beta_l | \sigma^2_{0l}, \pi_l)]
\end{aligned}
$$

### Coordinate ascent in the approximate ELBO

With $\psi_{li} := {\bf x}_i^T \beta_l$ and $\psi_i = {\bf x}_i^T \beta = \sum {\bf x}_i^T \beta_l = \sum_l \psi_{li}$ and $\psi_{-li} = \psi_i - \psi_{li}$.

Consider  the approximate ELBO just as a function of $q_l$ 

$$
\begin{aligned}
\hat F_{\text{SuSiE},l}(q_l| A, X,  q_{-l}, \sigma^2_{0l}, \pi_l) 
= \mathbb E_{q_l} \left[
    \mathbb E_{q_{-l}} \left[ 
      \sum_i \phi_{{\bf a}_i}(\psi_{li} + \psi_{-li})
    \right] 
\right] - KL \left[p(\beta_l) || q(\beta_{-l})\right] + \kappa
\end{aligned}
$$

Where $\kappa$ is a constant that does not depend on $q_l$ (note: it does depend on $\{\sigma^2_{0l}\}, \{\pi_l\}$, but they are not written in the arguments to $\hat F_{\text{SuSiE},l}$. We will need to evaluate $\mathbb E_{q_{-l}}[\phi_{\bf a_i}(\psi_{li} + \psi_{-li})]$ for each $i$. Focusing on a single observation

$$
\mathbb E_{q{-l}} \left[\phi_{\bf a_i}(\psi_{li} + \psi_{-li})\right]
= \mathbb E_{q_{-l}} \left[\phi_{{\bf b}({\bf a}, \psi_{-li})}(\psi_{li})\right] = \phi_{{\bf c}({\bf a_i}, q_{-l})}(\psi_{li})
$$

We will write ${\bf b}_{li} = {\bf b}({\bf a}, \psi_{-li})$ and ${\bf c}_{li} = {\bf c}({\bf a_i}, q_{-l})$, Indicating that they are the coefficient for an $M$ degree polynomial in $\psi_{li}$.


Revisiting the approximate ELBO we see

$$
\begin{aligned}
\hat F_{\text{SuSiE},l}(q_l| A, X,  q_{-l}) 
& = \mathbb E_{q_l} \left[
    \mathbb E_{q_{-l}} \left[ 
      \sum_i \phi_{{\bf a}_i}(\psi_{li} + \psi_{-li})
    \right] 
\right] - KL \left[q(\beta_l) || p(\beta_{-l} | \sigma^2_{0l}, \pi_l)\right] + \kappa \\
&= \left(\mathbb E_{q_l} \left[
  \sum_i \phi_{{\bf c}_{li}}(\psi_{li})
\right] - KL \left[q(\beta_l) || p(\beta_{-l} | \sigma^2_{0l}, \pi_l)\right] \right)  + \kappa \\
&= \hat F_{\text{SER}}(q_l | C_{l}, X, \sigma^2_{0l}, \pi_l) + \kappa
\end{aligned}
$$

We can see that with respect to the $q_l$ the SuSiE approximate ELBO is equal, up to a constant, to the SER approximate ELBO, for a new set of coefficients $C_l$, which depend on $q_{-l}$. It follows that the coordinate ascent update for $q_l$ is achieved by fitting the polynomial approximate SER with coefficients $C_l$. 

This is analagous to how we fit linear SuSiE by a sequence of SERs: we can fit the polynomial approximate SuSiE with a sequence of polynomial approximate SERs. Rather than "residualizing", we fit the polynomial approximate SER with coefficients $C_l$.

The crux of this approach is having a fast way to compute $C_l$. 


### Update Rule

At iteration $t$ we have the current variational approximation 

$$
q^{(t)} = \prod q_l^{(t)}
$$

Define 
$$
{\bf c}_i^{(0)} := \mathcal M(\psi_i, q^{(0)}) {\bf a}_i
$$

Note that $\mathcal M(\psi_i, q^{(0)})$ gives the expected shift matrix that removes the entire linear prediction, so that ${\bf c}_{i0}^{(0)} = \phi_{\bf c_i^{(0)}}(0) = \mathbb E_{q^{(0)}}[\log p(y_i | \psi_i)]$.

At iteration $t$, our coordinate ascent updates require ${\bf c}_{li}^{(t)} = \mathcal M(\psi_{li}, q_{-l}^{(t)})$. However, we know that:

$$
{\bf c}_{i}^{(t)} = \mathcal M(\psi_{li}, q_l^{(t)}){\bf c}_{li}^{(t)} \implies {\bf c}_{li}^{(t)} = \mathcal M(\psi_{li}, q_l^{(t)})^{-1}{\bf c}_{i}^{(t)}
$$

We can use $C_l^{(t)}$ to compute $q_l^{(t+1)}$ and then

$$
{\bf c}_{i}^{(t)} \leftarrow \mathcal M(\psi_{li}, q_l^{(t + 1)}){\bf c}_{li}^{(t)}
$$

so, by solving a triangular systems, and multiplying by upper triangular matrices $\mathcal M$ we can "efficiently" move between the coefficient representations needed for each SER update. I worry that $\mathcal M_l$ may be poorly conditioned resulting in numerical instability, but I have not seen it in toy examples yet.

### Algorithm 

Initialize $C^{(0)} = A$ and $q^{(0)} = \prod q_l^{(0)}$ such that $\mathbb E[\psi_l^k] = 0\;\; \forall l, k$.

For $t = 1, 2, \dots$:

  1. $C^{(t)} = C^{(t-1)}$

  1. For each $l \in [L]$:
  
      1. Compute $C_l^{(t)}$ by ${\bf c}_{li}^{(t)} \leftarrow \mathcal M(\psi_{li}, q_l^{(t-1)})^{-1} {\bf c}_i^{(t)}$ for $i \in [n]$
    
      1. $q_l^{(t)} \leftarrow \text{SER}(C_l^{(t)}, X, \sigma_{0l}^2, \pi_l)$
    
      1. Update $C^{(t)}$ by ${\bf c}_i^{(t)} \leftarrow \mathcal M(\psi_{li}, q^{(t)}_l) {\bf c}^{(l)}_i$

### Complexity

Let's break down the complexity of the inner loop (that is, updating $q_l$ and the requisite computations for the next run of the loop).

**Updating coefficients**: Translating between polynomials $O(M^2)$ to construct $\mathcal M$ and carry out matrix-vector multiplication or solve the triangular system. This is per observations, so $O(nM^2)$

**Fitting SER** Once we computed the coefficients we update the SER in $O(Mnp)$ (we just sum the coefficients to construct the polynomial posterior for each variable). Then $O(pM^3)$ to perform root finding and make the Gaussian approximation for $q_l$.

**Computing moments** We need to evaluate the moments $\mathbb E[\psi_{li}^k]$ for $k=1, \dots, M-1$. This is fast if $q_l$ is Gaussian but slower if not... $O(nM + ?)$, where $?$ is for computing $\mu_l^k = \mathbb E_{q_l}[\beta_l]$ $O(npM)$?

Total $O(M^2n + Mnp + M^3p)$


