# bayesian-modeling-mdc
# 🖥️ PHYS5001 Mock Data Challenge – Advanced Data Analysis

This repository contains my complete solution to the **Mock Data Challenge** for the University of Glasgow module *PHYS5001: Advanced Data Analysis*. The challenge involves fitting statistical models, performing Bayesian inference, and using Metropolis-Hastings MCMC techniques on synthetic datasets provided as `MDC1.txt` and `MDC2.txt`.

---

## 📂 Repository Structure

- `code/`: Python scripts for each step
- `data/`: Challenge input files
- `results/`: Output plots and tables
- `report.md`: Interpretation and commentary
- `README.md`: Project overview (this file)

---

## 📊 Part 1 – Linear Model: *y = a + bx*

### 🔹 Step 1.1: Ordinary Least Squares (OLS)
- Fitted the linear model to `MDC1.txt`
- Computed:
  - Least squares estimators for parameters *a* and *b*
  - Gaussian noise variance estimate
  - Errors and covariance of estimators
- Generated data plot with fitted line

### 🔹 Step 1.2: Maximum Likelihood Estimation
- Reformulated as a likelihood problem
- Evaluated:
  - Log-likelihood on a grid of *(a, b)* values
  - Chi-squared and minimum values
  - Δχ² array
- Plotted 2D Bayesian credible regions at:
  - 68.3%, 95.4%, 99.73%
- Listed best-fit values for *a*, *b* across confidence levels

### 🔹 Step 1.3: MCMC with Metropolis Algorithm
- Implemented a custom MCMC sampler
- Set priors and proposal distributions
- Sampled posteriors of *a*, *b*
- Derived marginal means, uncertainties, and covariances
- Created a corner plot of 1D and 2D posterior distributions

---

## 📐 Part 2 – Quadratic Model: *y = a + bx + cx²*

### 🔹 Step 2.1: Bayesian Inference via MCMC
- Fitted quadratic model to `MDC2.txt`
- Sampled posterior distributions for *a*, *b*, *c*
- Estimated Gaussian noise variance
- Created full corner plot with 1D & 2D posteriors

### 🔹 Step 2.2: Bayesian Model Comparison
- Computed marginal likelihoods for:
  - Linear model (from Part 1)
  - Quadratic model (from Part 2)
- Used `logsumexp` for precision during integration
- Calculated Bayes Factor to compare model suitability
- Explained assumptions behind prior ranges

---

## 🔧 Tools & Libraries
- Python: NumPy, SciPy, Matplotlib
- MCMC implementation from scratch
- Corner plots using `corner` or custom code
- `logsumexp` for stable likelihood integration

---

## 🧾 Interpretation & Insights
This challenge demonstrates the use of both frequentist and Bayesian approaches to parameter estimation. Starting from linear OLS fitting to advanced posterior sampling via MCMC, the project concludes with model comparison through Bayes Factor analysis.

---

## 📌 Author
**Nandadulal Ghosh**  
MSc candidate at the University of Glasgow (2023–2024)  
Strong foundation in statistical physics, data modeling, and Python-based analysis.

---

Feel free to fork, explore, and use this repository as a reference for similar analysis challenges in physics or beyond!
