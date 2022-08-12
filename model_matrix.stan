data {

  int <lower = 0> N; // Defining the number of defects in the test dataset
  array [N] int <lower = 0, upper = 1> y ; // A variable that describes whether each defect was detected [1]or not [0]
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix

}

parameters {
  vector [K] beta;
}

model {
     // Prior models for the unobserved parameters
    beta ~ normal(1, 10);

    y ~ bernoulli_logit(x * beta);

}
