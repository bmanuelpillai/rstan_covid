functions {
  // SIR model ODE
  vector sir_ode(real t, vector y, real beta, real gamma) {
    real S = y[1]; // Susceptible fraction
    real I = y[2]; // Infected fraction
    real R = y[3]; // Recovered fraction

    vector[3] dydt;
    dydt[1] = -beta * S * I;           // dS/dt
    dydt[2] = beta * S * I - gamma * I; // dI/dt
    dydt[3] = gamma * I;                // dR/dt

    return dydt;
  }
}

data {
  int<lower=1> T;                // Number of time points
  real t0;                        // Initial time
  array[T] real ts;               // Time points
  array[T] int cases;             // Observed daily cases
  real population;                // Total population
  real<lower=0> S0_frac;          // Initial susceptible fraction
  real<lower=0> I0_frac;          // Initial infected fraction
}

transformed data {
  vector[3] y0;                   // Initial states (fractions)
  y0[1] = S0_frac;
  y0[2] = I0_frac;
  y0[3] = 1.0 - S0_frac - I0_frac; // Ensure fractions sum to 1
}

parameters {
  real<lower=0> beta;             // Transmission rate
  real<lower=0> gamma;            // Recovery rate
  real<lower=0, upper=1> rho;     // Reporting rate
  real<lower=0> phi;              // Overdispersion for Neg-Binomial
}

transformed parameters {
  // Solve ODE to get predicted states over time
  array[T] vector[3] y_hat = ode_rk45(sir_ode, y0, t0, ts, beta, gamma);

  array[T] real incidence; // Predicted daily new infections
  for (t in 1:T) {
    if (t == 1)
      // Change in Susceptibles from t0 to t1
      incidence[t] = fmax(1e-6, (y0[1] - y_hat[t][1]) * population);
    else
      // Change in Susceptibles from t-1 to t
      incidence[t] = fmax(1e-6, (y_hat[t-1][1] - y_hat[t][1]) * population);
  }
}

model {
  // Priors
  beta ~ lognormal(log(0.4), 0.5);
  gamma ~ lognormal(log(0.14), 0.2);
  rho ~ beta(2, 5);
  phi ~ exponential(1);

  // Likelihood
  for (t in 1:T) {
    // mu is the reported incidence:
    // reporting rate * true new infections
    real mu = rho * incidence[t];
    cases[t] ~ neg_binomial_2(mu, phi);
  }
}

generated quantities {
  real R0 = beta / gamma;          // Basic reproduction number
  array[T] int y_rep;              // Posterior predictive checks

  for (t in 1:T) {
    real mu = rho * incidence[t];
    y_rep[t] = neg_binomial_2_rng(mu, phi);
  }
}


// Source of Priors:
// The median with 95% CI of R0 of COVID-19 was about 2.28 (2.06,2.52) (https://www.sciencedirect.com/science/article/pii/S1201971220300916)
// Assuming an infectious period of about 7 days, gamma ~ 1/7 = 0.14(https://www.nhs.uk/conditions/covid-19/how-to-avoid-catching-and-spreading-covid-19/)
// Thus, beta = R0 * gamma ~ 2.28 * 0.14 = 0.32
// We use a lognormal prior for beta with mean around 0.4 to allow some flexibility.
// The reporting rate rho is assumed to be low, so a Beta(2,5) prior.
