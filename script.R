library(tidyverse)
library(rstan)
library(ggdark)

#################### INTUITION OF BAYESIAN ######################
# this simple binomial toin coss example shows many features of a Bayesian approach to statistics

n <- 200
set.seed(1234)
outcomes <- cumsum(rbinom(n, 1, 0.5))
samples <- 1:n
p_hat <- outcomes / samples
p <- seq(0, 1, length = n)
sd_p <- sqrt(0.5 * (1 - 0.5) / samples)

# Under a frequentist perspective, we might see that the more coin flips we make, the closer our sample estimate converges to the true proportion of 0.5 and the smaller the sd of p.

ggplot() +
    geom_line(aes(samples, p_hat)) +
    geom_ribbon(aes(samples,
        ymin = p_hat - sd_p,
        ymax = p_hat + sd_p
    ), alpha = 0.3) +
    geom_hline(yintercept = 0.5, linetype = 2) +
    dark_theme_bw()
ggsave("./plots/plot1.png", width = 8, height = 6)

# but another way of seeing it is that the probabilty density function of p changes as we observe more data. The probability density starts wide, and gradually concentrates around the true value of p as more coin tosses are observed.

p <- seq(0, 1, length = n)
tibble(samples, outcomes) %>%
    group_by(samples, outcomes) %>%
    expand_grid(p) %>%
    rowwise() %>%
    mutate(log_lik = sum(log(dbinom(outcomes, samples, p)))) %>%
    mutate(lik = exp(log_lik)) %>%
    group_by(samples) %>%
    mutate(lik = lik / sum(lik)) %>%
    ggplot() +
    geom_line(aes(p, lik, group = samples, color = "white"), alpha = 0.2) +
    guides(color = "none") +
    geom_vline(xintercept = 0.5, linetype = 2) +
    theme_bw() +
    dark_theme_gray()

ggsave("./plots/plot2.png", width = 8, height = 6)

# the other thing to notice, even in this example is the computational intensity of the Bayesian approach in which we estimated the probability density of for all hypothetical values of p(x).

# in practice, we do not need to estimate all values of p(x), there are clever ways.

################### FIT LINEAR MODEL #######################
####### frequentist ########################
data(iris)
d <- as.data.frame(iris)

p <- ggplot(d, aes(Sepal.Length, Petal.Length, color = Species)) +
    geom_point() +
    dark_theme_grey()
p
ggsave("./plots/plot3.png", width = 8, height = 6)


m1 <- glm(Petal.Length ~ Sepal.Length + Species,
    family = gaussian,
    data = d
)

m1_pred <- predict(m1, se.fit = TRUE) %>%
    as_tibble() %>%
    mutate(
        lower = fit - 1.96 * se.fit,
        upper = fit + 1.96 * se.fit
    )
p +
    geom_line(aes(y = m1_pred$fit)) +
    geom_ribbon(aes(ymin = m1_pred$lower, ymax = m1_pred$upper, fill = Species),
        alpha = 0.3,
        color = NA
    )

ggsave("./plots/plot4.png", width = 8, height = 6)

summary(m1)


############ FIT BAYESIAN MODEL WITH STAN ###############

mat <- model.matrix(m1)

lookup(dnorm)

stan_data <- list(
    N = nrow(mat),
    y = d$Petal.Length,
    K = ncol(mat),
    X = mat
)

m2 <- stan(
    file = "lm_model.stan",
    data = stan_data,
    warmup = 500,
    iter = 3000,
    chains = 4,
    cores = 4,
    thin = 1,
    seed = 123
)

print(m2, probs = c(0.025, 0.975))

e <- rstan::extract(m2)
str(e)

dim(e$beta)
dim(mat)
pred_post <- e$beta %*% t(mat)
pred <- apply(pred_post, 2, function(x) quantile(x, c(0.025, 0.5, 0.975)))
m2_pred <- d %>%
    transmute(
        lower = pred[1, ],
        mu    = pred[2, ],
        upper = pred[3, ]
    )
p +
    geom_line(aes(y = m2_pred$mu)) +
    geom_ribbon(aes(ymin = m2_pred$lower, ymax = m2_pred$upper, fill = Species),
        alpha = 0.3,
        color = NA
    )

ggsave("./plots/plot5.png", width = 8, height = 6)

############ BINOMIAL LOGIT REGRESSION ################
############ frequentist logit ###################

# With the previous knowledge, it is relatively straigtforward to modify the approach for other models. Here we fit a binomial regression.

lookup("rbern")

d2 <- d %>%
    mutate(setosa = as.integer(Species == "setosa"))

p2 <- d2 %>%
    ggplot(aes(Sepal.Length, setosa)) +
    geom_point(aes(color = Species)) +
    dark_theme_grey()
p2 
ggsave("./plots/plot6.png", width = 8, height = 6)

glm2 <- glm(setosa ~ Sepal.Length, data = d2, family = binomial(link = "logit"))


ilogit = function(x) exp(x)/(1+exp(x))

m2_pred <- predict(glm2, se.fit = TRUE) %>%
    as_tibble() %>%
    mutate(
        lower = ilogit(fit - 1.96 * se.fit),
        upper = ilogit(fit + 1.96 * se.fit)
    ) %>% 
    mutate(fit = ilogit(fit))


p2 +
    geom_line(aes(y = m2_pred$fit)) +
    geom_ribbon(aes(ymin = m2_pred$lower, ymax = m2_pred$upper),
        alpha = 0.3,
        color = NA
    )

ggsave("./plots/plot7.png", width = 8, height = 6)

summary(glm2)


############### BAYESIAN LOGIT ###############
mat <- model.matrix(glm2)

stan_data2 <- list(
    N = nrow(mat),
    y = d2$setosa,
    K = ncol(mat),
    X = mat
)

m2 <- stan(
    file = "logit_model.stan",
    data = stan_data2,
    warmup = 500,
    iter = 3000,
    chains = 4,
    cores = 4,
    thin = 1,
    seed = 123
)

print(m2, probs = c(0.025, 0.975))

e <- rstan::extract(m2)
str(e)

dim(e$beta)
dim(mat)

pred_post <-  ilogit(e$beta %*% t(mat))
pred <- apply(pred_post, 2, function(x) quantile(x, c(0.025, 0.5, 0.975)))
m2_pred <- d2 %>%
    transmute(
        lower = pred[1, ],
        mu    = pred[2, ],
        upper = pred[3, ]
    )
p2 +
    geom_line(aes(y = m2_pred$mu)) +
    geom_ribbon(aes(ymin = m2_pred$lower, ymax = m2_pred$upper),
        alpha = 0.3,
        color = NA
    )

ggsave("./plots/plot8.png", width = 8, height = 6)
