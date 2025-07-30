#step 1.1
import numpy as np
import matplotlib.pyplot as plt

# Load Data
# Due to different path of installation and stored data pathway needed to be established
data = np.loadtxt('E:\\ADA\\Mock data challenge data and resources-20240329\\MDC1.txt')
x = data[:, 0]
y = data[:, 1] 

# Perform the fit
A = np.vstack([x, np.ones(len(x))]).T
b, a = np.linalg.lstsq(A, y, rcond=None)[0]

# Estimate the variance of the noise
residuals = y - (a + b * x)
noise_variance = np.var(residuals)

# Plot the data and the fit line
plt.plot(x, y, 'o', label='Original data')
plt.plot(x, a + b * x, 'r', label='Fitted line')
plt.legend()
plt.show()

#step 1.2
from scipy.optimize import minimize

# Define the likelihood function
def likelihood(params):
    a, b = params
    residuals = y - (a + b * x)
    return np.sum(residuals**2)

# Optimize the likelihood function
initial_guess = [1, 1]
result = minimize(likelihood, initial_guess)
a_mle, b_mle = result.x

#step 1.3
def metropolis(start, proposal, niter, nburn=0, **kwargs):
    current = start
    post = -likelihood(start)
    samples = np.zeros((niter+nburn, len(start)))

    for i in range(niter+nburn):
        proposed = proposal(current)
        post_proposed = -likelihood(proposed)

        if np.random.rand() < np.exp(post_proposed - post):
            current = proposed
            post = post_proposed

        samples[i] = current

    return samples[nburn:]

# Define the proposal density
def proposal(current):
    return np.random.normal(current, [0.5, 0.5])

# Run the Metropolis algorithm
samples = metropolis(initial_guess, proposal, 10000, nburn=1000)

#step 2.1
# Define the likelihood function for the quadratic model
def likelihood_quad(params):
    a, b, c = params
    residuals = y - (a + b * x + c * x**2)
    return np.sum(residuals**2)

# Run the Metropolis algorithm
samples_quad = metropolis([1, 1, 1], proposal, 10000, nburn=1000)

#step 2.2
from scipy.special import logsumexp

# Define a grid of parameter values
a_values = np.linspace(-10, 10, 100)
b_values = np.linspace(-10, 10, 100)

# Compute the log likelihood values on the grid
log_likelihoods_linear = np.array([[likelihood([a, b]) for a in a_values] for b in b_values])
log_likelihoods_quad = np.array([[likelihood_quad([a, b, 1]) for a in a_values] for b in b_values])

# Compute the marginal likelihoods
marginal_likelihood_linear = logsumexp(-log_likelihoods_linear)
marginal_likelihood_quad = logsumexp(-log_likelihoods_quad)

# Compute the Bayes factor
bayes_factor = marginal_likelihood_quad - marginal_likelihood_linear

