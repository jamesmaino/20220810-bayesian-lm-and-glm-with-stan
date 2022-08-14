data {
  int <lower = 0> N; 
  int<lower=0,upper=1> y[N];
  int<lower=0> K;   
  matrix[N, K] X;  
}

parameters {
  vector [K] beta;
}

model {
    y ~ bernoulli_logit(X*beta);
}
