functions {
  // SIR ODE Model
  vector sir_ode(real t, vector y, real beta, real gamma) {
    real S = y[1]; // Susceptible
    real I = y[2]; // Infected
    real R = y[3]; // Recovered

    vector[3] dydt;
    dydt[1] = -beta * S * I;            // dS/dt
    dydt[2] = beta * S * I - gamma * I; // dI/dt
    dydt[3] = gamma * I;                // dR/dt

    return dydt;
  }
}

data {
  int<lower=1> T; // Number of time points
  real t0; // Initial time
  array[T] real ts; // Time points
  array[T] int cases; // Observed cases
  real population; // Total population
  real<lower=0> S0_frac; // Initial susceptible fraction
  real<lower=0> I0_frac; // Initial infected fraction
}

transformed data {
  vector[3] y0; // Initial States
  y0[1] = S0_frac;
  y0[2] = I0_frac;
  y0[3] = 0.0;
}

parameters {
  real<lower=0> beta; // Transmission rate
  real<lower=0> gamma; // Recovery rate
  real<lower=0, upper=1> rho; // Reporting rate
  real<lower=0> phi; // Overdispersion parameter
}

transformed parameters {
  array[T] vector[3] y_hat;
  array[T] real incidence;

  y_hat = ode_rk45(sir_ode, y0, t0, ts, beta, gamma);

  for (t in 1:T) {
    if (t == 1)
      incidence[t] = fmax(0.0, (y0[1] - y_hat[t][1]) * population);
    else
      incidence[t] = fmax(0.0, (y_hat[t-1][1] - y_hat[t][1]) * population);
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.4), 0.75);
  gamma ~ normal(1/7.0, 0.5);
  rho ~ beta(2, 8);
  phi ~ normal(0, 5);

  // Likelihood
  for (t in 1:T) {
    real mu = rho * incidence[t] + 1e-6;
    cases[t] ~ neg_binomial_2(mu, phi);
  }
}

generated quantities {
  real R0 = beta / gamma;
  array[T] int y_rep;

  for (t in 1:T) {
    real mu = rho * incidence[t] + 1e-6;
    y_rep[t] = neg_binomial_2_rng(mu, phi);
  }
}
