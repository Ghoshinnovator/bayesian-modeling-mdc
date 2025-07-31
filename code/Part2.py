#step2.1
import numpy as np
import matplotlib.pyplot as plt

# Load Data
# Due to different path of installation and stored data, pathway needed to be established
data = np.loadtxt('E:\\ADA\\Mock data challenge data and resources-20240329\\MDC2.txt')
x = data[:, 0]
y = data[:, 1]

# Define the proposal density
def proposal(a, b, c):
    return [np.random.normal(a, 1), np.random.normal(b, 1), np.random.normal(c, 1)]

def likelihood(params, x, y):
    a, b, c = params
    y_model = a + b * x + c * x**2
    residuals = y - y_model
    sigma = np.std(residuals)  # standard deviation of the residuals
    likelihood = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return likelihood

# Initialize the MCMC
a_current, b_current, c_current = 0, 0, 0  # adjust as needed
samples = []

# Run the MCMC
for i in range(10000):  # adjust as needed
    a_proposal, b_proposal, c_proposal = proposal(a_current, b_current, c_current) #a_current, b_current, and c_current are passed as a list to the likelihood function
    likelihood_current = likelihood([a_current, b_current, c_current], x, y)
    likelihood_proposal = likelihood([a_proposal, b_proposal, c_proposal], x, y)
    
    # Acceptance probability
    log_p_accept = likelihood_proposal - likelihood_current

    # Accept or reject the proposal
    if np.log(np.random.rand()) < log_p_accept:
     a_current, b_current, c_current = a_proposal, b_proposal, c_proposal

# Accept or reject the proposal
if np.log(np.random.rand()) < log_p_accept:
    a_current, b_current, c_current = a_proposal, b_proposal, c_proposal

# Append the current parameters to the samples list
samples.append([a_current, b_current, c_current])

samples = np.array(samples)

# Plot histograms of the parameter values
plt.figure(figsize=(12, 3))
for i, label in enumerate(['a', 'b', 'c']):
    plt.subplot(1, 3, i+1)
    plt.hist(samples[:, i], bins=50)
    plt.title(label)
plt.tight_layout()
plt.show()

#step 2.2
# Define the likelihood function for the linear model
def linear_likelihood(a, b, x, y):
    predicted_y = a + b*x
    residuals = y - predicted_y
    return -0.5 * np.sum(residuals**2)

# Define the likelihood function for the quadratic model
def quadratic_likelihood(a, b, c, x, y):
    predicted_y = a + b*x + c*x**2
    residuals = y - predicted_y
    return -0.5 * np.sum(residuals**2)

# Extract a_samples, b_samples, and c_samples from samples
a_samples = samples[:, 0]
b_samples = samples[:, 1]
c_samples = samples[:, 2]

# Compute the log likelihoods for the linear model
linear_log_likelihoods = linear_likelihood(a_samples, b_samples, x, y)

# Compute the log likelihoods for the quadratic model
quadratic_log_likelihoods = quadratic_likelihood(a_samples, b_samples, c_samples, x, y)


# Compute the marginal likelihoods using the logsumexp function
from scipy.special import logsumexp
linear_marginal_likelihood = logsumexp(linear_log_likelihoods)
quadratic_marginal_likelihood = logsumexp(quadratic_log_likelihoods)

# Compute the Bayes factor
bayes_factor = linear_marginal_likelihood - quadratic_marginal_likelihood

# Print the results
print(f"Linear Marginal Likelihood: {linear_marginal_likelihood}")
print(f"Quadratic Marginal Likelihood: {quadratic_marginal_likelihood}")
print(f"Bayes Factor: {bayes_factor}")


