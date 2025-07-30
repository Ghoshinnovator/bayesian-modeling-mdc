# Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import emcee

# Load Data
# Due to different path of installation and stored data, establish the pathway
data = np.loadtxt('E:\\ADA\\Mock data challenge data and resources-20240329\\MDC1.txt')
x = data[:, 0]
y = data[:, 1]

# Step 1.1: Ordinary Least Squares Fitting
def model(x, a, b):
    return a + b * x

popt, pcov = curve_fit(model, x, y)     # Use the curve_fit function to fit the model function to data
a, b = popt                             # Unpack the values of a and b
a_err, b_err = np.sqrt(np.diag(pcov))   # Calculate the standard errors of the parameters a and b
residuals = y - model(x, *popt)         # Calculate the residuals
noise_variance = np.var(residuals)      # Calculate the varience of residuals

# Plotting the data and the least squares fit line
plt.scatter(x, y, label='Data')
plt.plot(x, model(x, *popt), color='red', label=f'Fit: y = {a:.2f} + {b:.2f}x')
plt.legend()
plt.show()

# Step 1.2: Maximum Likelihood Estimation
# Define the likelihood function
def log_likelihood(theta, x, y):
    a, b = theta
    model = a + b * x
    sigma2 = np.sum((y - model) ** 2) / len(y)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# Grid of values for a and b
# Spanning three standard errors on either sides
a_values = np.linspace(a - 3*a_err, a + 3*a_err, 100)
b_values = np.linspace(b - 3*b_err, b + 3*b_err, 100)

# Compute the log likelihood on the grid
log_likelihood_values = np.empty((100, 100))
for i in range(100):
    for j in range(100):
        log_likelihood_values[i, j] = log_likelihood((a_values[i], b_values[j]), x, y)

# Compute chi-squared values and find the minimum
chi_squared_values = -2 * log_likelihood_values
min_chi_squared = np.min(chi_squared_values)

# Compute Delta chi-squared values
delta_chi_squared_values = chi_squared_values - min_chi_squared

# Step 1.3: Metropolis Algorithm
# Define the prior, likelihood and posterior
def log_prior(theta):
    a, b = theta
    if -10 < a < 10 and -10 < b < 10:
        return 0
    return -np.inf

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

# Initialize the MCMC sampler
ndim = 2
nwalkers = 50
p0 = [np.random.rand(ndim) for i in range(nwalkers)]         # Initialize starting points of the walkers in the parameter space
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y))

# Run the MCMC sampler
sampler.run_mcmc(p0, 1000)

# Extract the samples
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Compute the mean values, errors and covariance of the parameters
mean_a, mean_b = np.mean(samples, axis=0)
std_a, std_b = np.std(samples, axis=0)
cov_ab = np.cov(samples, rowvar=False)

# Plot the results
import corner
fig = corner.corner(samples, labels=["a", "b"], truths=[mean_a, mean_b])
plt.show()

