# Plots for report

library(dplyr)
library(ggplot2)

sigmoid <- function(x){1/(1 + exp(-x))}

simulate <- function(n, beta0=0, beta=1){
  x <- rnorm(n)
  logit <- beta0 + beta*x
  y <- rbinom(n, 1, sigmoid(logit))
  return(list(x=x, y=y, beta0=beta0, beta=beta))
}

fit_glm <- function(sim){
  fit <- with(sim, glm(y ~ x + 1, family = 'binomial'))
  tmp <- unname(summary(fit)$coef[2,])
  betahat <- tmp[1]
  shat2 <- tmp[2]^2
  z <- with(sim, (beta - betahat)/sqrt(shat2))
  
  null_loglik <- with(
    sim, dbinom(sum(y), length(y), mean(y), log = T))
  asymptotic_null_loglik <- dnorm(betahat, 0, sqrt(shat2), log=T)
  
  
  fit_summary <- summary(fit)
  mle_null_llr <- with(fit_summary, 0.5 * (null.deviance - deviance))
  
  res <- list(
    intercept = unname(fit$coefficients[1]),
    betahat = betahat,
    shat2 = shat2,
    z = z,
    null_loglik = null_loglik,
    mle_null_llr = mle_null_llr,
    asymptotic_null_loglik = asymptotic_null_loglik
  )
  return(res)
}

compute_log_abf1 <- function(betahat, shat2, prior_variance){
  W <- prior_variance
  V <- shat2
  z <- betahat / sqrt(shat2)
  labf <- 0.5 * log(V /(V + W)) + 0.5 * z^2 * W /(V + W)
  return(labf)
}

compute_log_abf <- function(sim, prior_variance=1){
  labf <- sim %>% 
    fit_glm() %>% 
    {compute_log_abf1(.$betahat, .$shat2, prior_variance)}
  return(labf)
}

compute_log_labf <- function(sim, prior_variance=1){
  glm_fit <- fit_glm(sim)
  llabf <- glm_fit$mle_null_llr + 
    dnorm(glm_fit$betahat, 0, sqrt(glm_fit$shat2 + prior_variance), log=T) -
    dnorm(glm_fit$betahat, glm_fit$betahat, sqrt(glm_fit$shat2), log=T)
  return(llabf)
}

compute_vb_log_bf <- function(sim, prior_variance=1){
  vb <- with(sim, logisticsusie::fit_univariate_vb(x, y, tau0=1/prior_variance)) 
  log_vb_bf <- tail(vb$elbos, 1) - sum(dbinom(sim$y, 1, mean(sim$y), log=T))
  return(log_vb_bf)
}

compute_exact_log_bf <- function(sim, prior_variance=1){
  glm_fit <- fit_glm(sim)
  f <- function(b){
    logits <- sim$x * b + glm_fit$intercept
    sum(dbinom(x=sim$y, size=1, prob=sigmoid(logits), log = T)) + 
      dnorm(b, 0, sqrt(prior_variance), log = T)
  }
  max <- f(glm_fit$betahat)
  
  f2 <- function(b){purrr::map_dbl(b, ~exp(f(.x) - max))}
  
  lower <- glm_fit$betahat - sqrt(glm_fit$shat2) * 6
  upper <- glm_fit$betahat + sqrt(glm_fit$shat2) * 6
  quadres <- cubature::pcubature(f2, lower, upper, tol = 1e-10)
  log_bf <- (log(quadres$integral) + max) - sum(dbinom(sim$y, 1, mean(sim$y), log=T))
  return(log_bf)
}

bf_comparison <- function(sim, prior_variance){
  a <- sim %>% fit_glm() %>% compute_log_abf(prior_variance=prior_variance)
  b <- sim %>% compute_vb_log_bf(prior_variance=prior_variance)
  c <- sim %>% compute_exact_log_bf(prior_variance=prior_variance)
  tibble(log_abf = a, log_vbf = b, log_bf = c)
}

bf_comparison2 <- function(...){
  sim <- simulate(...)
  a <- sim %>% fit_glm() %>% compute_log_abf(prior_variance=1)
  b <- sim %>% compute_vb_log_bf(prior_variance=1)
  c <- sim %>% compute_exact_log_bf(prior_variance=1)
  tibble(log_abf = a, log_vbf = b, log_bf = c)
}

subset_sim <- function(sim, m){
  sim2 <- sim
  sim2$x <- head(sim2$x, m)
  sim2$y <- head(sim2$y, m)
  return(sim2)
}

compare_lrs <- function(sim, prior_variance=1, k=1, plot_correction=F){
  fit <- fit_glm(sim)
  fit_no_intercept <- fit_glm_no_intercept(sim)
  
  # plot ll
  ll <- function(b){with(sim, sum(y*x*b - log(1 + exp(x*b))))}
  ll_vec <- function(b){purrr::map_dbl(b, ~ll(.x))}
  asymptotic_ll <- function(b){dnorm(b, mean=fit$betahat, sd=sqrt(fit$shat2), log=T)}
  
  ll0 <- ll(0)
  asymptotic_ll0 <- asymptotic_ll(0)
  
  ll_mle <- ll(fit$betahat)
  asymptotic_ll_mle <- asymptotic_ll(fit$betahat)
  
  mle = fit$betahat
  xs <- seq(mle - k* abs(mle), mle + k* abs(mle), by=0.1)
  
  diff = (ll_mle - asymptotic_ll_mle) - ll0
  ll_xs <- ll_vec(xs)
  asymptotic_ll_xs <- asymptotic_ll(xs)
  #ll_prior <- dnorm(xs, 0, sd = sqrt(prior_variance), log = T)
  
  ylim <- range(asymptotic_ll_xs - asymptotic_ll0, ll_xs - ll0)
  plot(xs,ll_xs - ll0, type='l', ylim = ylim,
       xlab= 'beta', ylab='Log-likelihood ratio')
  lines(xs, asymptotic_ll_xs - asymptotic_ll0, col='red')
  
  if(plot_correction){
    lines(xs, asymptotic_ll_xs - asymptotic_ll_mle + (ll_mle - ll0), col='blue')
  }
  abline(h=0, lty=3)
  abline(v=0, lty=3)
  abline(v=fit$betahat)
  
  sim %>% bf_comparison(1)
  
  laplace_lbf <- compute_log_labf(sim)
  wakefield_lbf <- compute_log_abf(sim)
  exact_lbf <- compute_exact_log_bf(sim)
  
  l1 <- paste0('Laplace (log BF = ', format(round(laplace_lbf, 3)), ')')
  l2 <- paste0('Wakefield (log BF = ', format(round(wakefield_lbf, 3)), ')')
  l3 <- paste0('Exact (log BF = ', format(round(exact_lbf, 3)), ')')
  
  legend('bottomright', 
         legend = c(l1, l2, l3), 
         col=c('blue', 'red', 'black'), lty = 1, inset=0.01)
}

# ABF vs BF error increases with BF:
# make a large simulation, and track BF vs ABF computed on nested
# subsets of the data
plot_log_bf_vs_log_abf <- function(){
  set.seed(2)
  n <- 2^15
  sim <- simulate(2^15, beta=1)
  
  m <- as.integer(cumprod(c(64, rep(sqrt(sqrt(2)), 36))))
  res <- purrr::map_dfr(m, ~ sim %>% subset_sim(.x) %>% bf_comparison(1))
  plot(res$log_bf, res$log_abf,
       xlab = 'log(BF)', ylab= 'log(ABF)'); 
  abline(0, 1, col='red')
}


plot_likelihood_ratio_approximations <- function(){
  set.seed(2)
  sim <- simulate(1000)
  compare_lrs(sim, prior_variance = 1, k=1.5, plot_correction = T)
}


fit_glm_ser <- function (X, y, o = NULL, prior_variance = 1, estimate_intercept = T, 
                         prior_weights = NULL, family = "binomial") {
  p <- ncol(X)
  betahat <- numeric(p)
  shat2 <- numeric(p)
  intercept <- rep(0, p)
  lr <- rep(0, p)
  if (is.null(o)) {
    o <- rep(0, length(y))
  }
  for (j in 1:p) {
    if (estimate_intercept) {
      log.fit <- glm(y ~ X[, j] + 1 + offset(o), family = family)
      intercept[j] <- unname(coef(log.fit)[1])
    }
    else {
      log.fit <- glm(y ~ X[, j] - 1 + offset(o), family = family)
    }
    log.fit.coef <- summary(log.fit)$coefficients
    betahat[j] <- ifelse(is.na(log.fit.coef[1 + estimate_intercept, 
                                            1]), 0, log.fit.coef[1 + estimate_intercept, 1])
    shat2[j] <- ifelse(is.na(log.fit.coef[1 + estimate_intercept, 
                                          2]), Inf, log.fit.coef[1 + estimate_intercept, 2]^2)
    lr[j] <- 0.5 * (log.fit$null.deviance - log.fit$deviance)
  }
  lbf <- dnorm(betahat, 0, sqrt(prior_variance + shat2), log = TRUE) - 
    dnorm(betahat, 0, sqrt(shat2), log = TRUE)
  lbf[is.infinite(shat2)] <- 0
  if (is.null(prior_weights)) {
    prior_weights <- rep(1/p, p)
  }
  maxlbf <- max(lbf)
  w <- exp(lbf - maxlbf)
  w_weighted <- w * prior_weights
  weighted_sum_w <- sum(w_weighted)
  alpha <- w_weighted/weighted_sum_w
  
  #corrected ABF
  alr_mle <- dnorm(betahat, betahat, sqrt(shat2), log=T) - 
    dnorm(betahat, 0, sqrt(shat2), log=T)
  lbf_corrected <- lbf - alr_mle + lr
  alpha_corrected <- exp(lbf_corrected - matrixStats::logSumExp(lbf_corrected))
  
  # posterior
  post_var <- 1/((1/shat2) + (1/prior_variance))
  post_mean <- (1/shat2) * post_var * betahat
  post_mean2 <- post_var + post_mean^2
  lbf_model <- maxlbf + log(weighted_sum_w)
  null_loglik <- sum(dbinom(y, 1, mean(y), log = T))
  loglik <- lbf_model + null_loglik
  
  res <- list(
    alpha = alpha,
    mu = post_mean,
    var = post_var,
    lbf = lbf,
    lr = lr,
    null_loglik = null_loglik,
    lbf_model = lbf_model,
    prior_variance = prior_variance, 
    loglik = loglik,
    betahat = betahat,
    shat2 = shat2,
    intercept = intercept,
    lbf_corrected = lbf_corrected,
    alpha_corrected = alpha_corrected
  )
  return(res)
}

ser_reorder_plot <- function(){
  set.seed(2)
  # "corrected" ABF-- laplace approximation
  sim <- logisticsusie::sim_ser()
  glm <- with(sim, fit_glm_ser(X, y))
  ser <- with(sim, logisticsusie::fit_uvb_ser(X, y))
  
  alr_mle <- with(glm, 
                  dnorm(betahat, betahat, sqrt(shat2), log=T) - 
                    dnorm(betahat, 0, sqrt(shat2), log=T))
  
  lbf_corrected <- with(glm, lbf - alr_mle + lr)
  alpha_corrected <- exp(lbf_corrected - matrixStats::logSumExp(lbf_corrected))
  
  plot(ser$alpha, pch=4, xlab='Variable', ylab ='Posterior inclusion probability (PIP)')
  points(glm$alpha, col='red')
  points(alpha_corrected, col='blue')
  legend('topright',
         legend=c('Exact', 'ABF', 'Laplace ABF'),
         col=c('black', 'red', 'blue'),
         pch = c(1, 1, 4), inset = 0.05)
}
