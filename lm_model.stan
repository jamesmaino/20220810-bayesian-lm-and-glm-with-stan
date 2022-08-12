data {
  int <lower = 0> N; 
  vector [N] y ; 
  int<lower=0> K;   
  matrix[N, K] X;  
}

parameters {
  vector [K] beta;
  real <lower=0> sigma;
}

model {
    y ~ normal_lpdf(beta*X, sigma);
}
