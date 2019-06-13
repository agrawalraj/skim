
import pystan
import numpy as np

# adapted from https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

def make_stan_gpr(squared=True):
	if squared:
		gp_fin_code = """
				data {
				  int<lower=1> N; // Number of data
				  int<lower=1> M; // Number of covariates
				  matrix[N, M] X;
				  vector[N] y;
				  // vector[M] X[N];
				  // Interaction global scale params
				  real<lower=0> c; // Intercept prior scale
				  real m0; // Expected number of large slopes
				}

				// slab_scale = 5, slab_df = 25 -> 8 divergences

				transformed data {
				  real slab_scale = 3;    // Scale for large slopes
				  real slab_scale2 = square(slab_scale);
				  real slab_df = 25;      // Effective degrees of freedom for large slopes
				  real half_slab_df = 0.5 * slab_df;
				  vector[N] mu = rep_vector(0, N);
				  // vector[M] X2[N] = square(X);
				  matrix[N, M] X2 = square(X);
				}

				parameters {
				  vector<lower=0>[M] lambda;
				  real<lower=0> m_base;
				  real<lower=0> eta_1_base;
				  real<lower=0> sigma; // Noise scale of response
				  real<lower=0> psi; // Interaction scale (selected ones)
				}

				transformed parameters {
				  real<lower=0> eta_2;
				  real<lower=0> alpha; // Prior variance on quadratic effect
				  real<lower=0> eta_1;
				  real<lower=0> m_sq; // Truncation level for local scale horseshoe
				  vector[M] kappa;
				  {
				    real phi = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
				    eta_1 = phi * eta_1_base; // eta_1 ~ cauchy(0, phi), global scale for linear effects

				    // m_sq ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
				    m_sq = slab_scale2 * m_base; // m^2
				    kappa = m_sq * square(lambda) ./ (m_sq + square(eta_1) * square(lambda));
				  }
				  eta_2 = square(eta_1) / m_sq * psi; // Global prior variance of interaction terms
				  alpha = 0; // No quadratic effects
				}

				model {
				  matrix[N, N] L_K;
				  matrix[N, N] K1 = diag_post_multiply(X, kappa) *  X';
				  matrix[N, N] K2 = diag_post_multiply(X2, kappa) *  X2';
				  matrix[N, N] K = .5 * square(eta_2) * square(K1 + 1.0) + (square(alpha) - .5 * square(eta_2)) * K2 + (square(eta_1) - square(eta_2)) * K1 + square(c) - .5 * square(eta_2);
				  
				  // diagonal elements
				  for (n in 1:N)
				    K[n, n] += square(sigma);

				  L_K = cholesky_decompose(K);

				  lambda ~ cauchy(0, 1);
				  eta_1_base ~ cauchy(0, 1);
				  m_base ~ inv_gamma(half_slab_df, half_slab_df);

				  sigma ~ normal(0, 2);
				  psi ~ inv_gamma(half_slab_df, half_slab_df);

				  y ~ multi_normal_cholesky(mu, L_K);
				}
				"""
	else:
		gp_fin_code = """
			data {
			  int<lower=1> N; // Number of data
			  int<lower=1> M; // Number of covariates
			  matrix[N, M] X;
			  vector[N] y;
			  // vector[M] X[N];
			  // Interaction global scale params
			  real<lower=0> c; // Intercept prior scale
			  real m0; // Expected number of large slopes
			}

			// slab_scale = 5, slab_df = 25 -> 8 divergences

			transformed data {
			  real slab_scale = 3;    // Scale for large slopes
			  real slab_scale2 = square(slab_scale);
			  real slab_df = 25;      // Effective degrees of freedom for large slopes
			  real half_slab_df = 0.5 * slab_df;
			  vector[N] mu = rep_vector(0, N);
			  // vector[M] X2[N] = square(X);
			  matrix[N, M] X2 = square(X);
			}

			parameters {
			  vector<lower=0>[M] lambda;
			  real<lower=0> m_base;
			  real<lower=0> eta_1_base;
			  real<lower=0> sigma; // Noise scale of response
			  real<lower=0> psi; // Interaction scale (selected ones)
			}

			transformed parameters {
			  real<lower=0> eta_2;
			  real<lower=0> alpha; // Prior variance on quadratic effect
			  real<lower=0> eta_1;
			  real<lower=0> m_sq; // Truncation level for local scale horseshoe
			  vector[M] kappa;
			  {
			    real phi = (m0 / (M - m0)) * (sigma / sqrt(1.0 * N));
			    eta_1 = phi * eta_1_base; // eta_1 ~ cauchy(0, phi), global scale for linear effects

			    // m_sq ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
			    m_sq = slab_scale2 * m_base; // m_sq^2
			    kappa = sqrt(m_sq * square(lambda) ./ (m_sq + square(eta_1) * square(lambda)));
			  }
			  eta_2 = eta_1 / sqrt(m_sq) * psi; // Global prior variance of interaction terms
			  alpha = 0; // No quadratic effects
			}

			model {
			  matrix[N, N] L_K;
			  matrix[N, N] K1 = diag_post_multiply(X, kappa) *  X';
			  matrix[N, N] K2 = diag_post_multiply(X2, kappa) *  X2';
			  matrix[N, N] K3 = diag_post_multiply(X, square(kappa)) *  X';
			  matrix[N, N] K = .5 * square(eta_2) * square(K1 + 1.0) + (square(alpha) - .5 * square(eta_2)) * K2 + (square(eta_1) - square(eta_2)) * K1 + square(c) - .5 * square(eta_2) - square(eta_1) * K_1 + square(eta_1) * K_3;
			  
			  // diagonal elements
			  for (n in 1:N)
			    K[n, n] += square(sigma);

			  L_K = cholesky_decompose(K);

			  lambda ~ cauchy(0, 1);
			  eta_1_base ~ cauchy(0, 1);
			  m_base ~ inv_gamma(half_slab_df, half_slab_df);

			  sigma ~ normal(0, 2);
			  psi ~ inv_gamma(half_slab_df, half_slab_df);

			  y ~ multi_normal_cholesky(mu, L_K);
			}
			"""
	return pystan.StanModel(model_code=gp_fin_code)

