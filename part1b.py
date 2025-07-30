#step1.1
import numpy as np
import matplotlib.pyplot as plt

# Assuming x and y are your data
x = np.array([...])  # replace with your x data
y = np.array([...])  # replace with your y data

# Perform the ordinary least squares fit
A = np.vstack([x, np.ones(len(x))]).T
b, a = np.linalg.lstsq(A, y, rcond=None)[0]

# Estimate the variance of the noise
residuals = y - (a + b*x)
noise_variance = np.var(residuals)

# Plot the data and the least squares fit line
plt.plot(x, y, 'o', label='Original data')
plt.plot(x, a + b*x, 'r', label='Fitted line')
plt.legend()
plt.show()

#step 1.2

# Define the likelihood function
def likelihood(a, b, x, y):
    predicted_y = a + b*x
    residuals = y - predicted_y
    return -0.5 * np.sum(residuals**2)

# Compute the log likelihood on a grid of values
a_values = np.linspace(-10, 10, 100)  # adjust as needed
b_values = np.linspace(-10, 10, 100)  # adjust as needed
log_likelihoods = np.zeros((len(a_values), len(b_values)))

for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        log_likelihoods[i, j] = likelihood(a, b, x, y)

# Find the minimum chi-squared value
min_chi_squared = -2 * np.max(log_likelihoods)

#step 1.3

# Define the proposal density
def proposal(a, b):
    return [np.random.normal(a, 1), np.random.normal(b, 1)]

# Initialize the MCMC
a_current, b_current = 0, 0  # adjust as needed
samples = []

# Run the MCMC
for i in range(10000):  # adjust as needed
    a_proposal, b_proposal = proposal(a_current, b_current)
    likelihood_current = likelihood(a_current, b_current, x, y)
    likelihood_proposal = likelihood(a_proposal, b_proposal, x, y)
    
    # Acceptance probability
    p_accept = min(1, np.exp(likelihood_proposal - likelihood_current))
    
    # Accept or reject the proposal
    if np.random.rand() < p_accept:
        a_current, b_current = a_proposal, b_proposal
    
    samples.append([a_current, b_current])

