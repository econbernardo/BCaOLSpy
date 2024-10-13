import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

class BiasCorrectedOLS:
    def __init__(self, df, dependent_var, independent_vars, alpha=0.05, verbose=True):
        """
        Initialize the BiasCorrectedOLS object.

        Parameters:
        - df: pandas.DataFrame containing the data for the regression.
        - dependent_var: str, name of the dependent variable.
        - independent_vars: list of str, names of the independent variables.
        - alpha: float, significance level for confidence intervals (default=0.05).
        - verbose: bool, if True, display progress using tqdm for bootstrapping and jackknife methods.
        """
        self.df = df
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.alpha = alpha
        self.verbose = verbose
        self.formula = f'{dependent_var} ~ {" + ".join(independent_vars)}'
        self.model = None
        self.coefs = None
        self.conf = None
        self.nobs = None
        self.bootstrap_distribution = None
        self.jackknife_distribution = None

    def run_regression(self):
        """
        Run an OLS regression and store the results.
        
        Returns:
        - coefs: estimated coefficients.
        - conf: confidence intervals for the estimated coefficients.
        - nobs: number of observations used in the regression.
        """
        model = sm.OLS.from_formula(self.formula, self.df, missing='drop').fit(cov_type='HC0')
        self.coefs = model.params
        self.conf = model.conf_int(alpha=self.alpha)
        self.nobs = model.nobs
        return self.coefs, self.conf, self.nobs

    def perform_bootstrap(self, n_sim, desc='Bootstrap Simulation'):
        """
        Perform bootstrap sampling to obtain the distribution of the OLS estimates.

        Parameters:
        - n_sim: int, number of bootstrap samples to run.
        - desc: str, description to display in tqdm for bootstrapping.

        Returns:
        - bootstrap_distribution: dictionary of bootstrapped coefficients for each variable.
        """
        coefs = {}
        progress = tqdm(range(n_sim), desc=desc) if self.verbose else range(n_sim)
        first_iter = True
        for _ in progress:
            bootstrap_df = self.df.sample(frac=1, replace=True)
            model = sm.OLS.from_formula(self.formula, bootstrap_df, missing='drop').fit()
            boot_coef = model.params
            if first_iter:
                coefs = {var: [] for var in boot_coef.index}
                first_iter = False
            for var in boot_coef.index:
                coefs[var].append(boot_coef[var])
        self.bootstrap_distribution = coefs
        return coefs

    def perform_jackknife(self):
        """
        Perform jackknife resampling to obtain the distribution of the OLS estimates.

        Returns:
        - jackknife_distribution: dictionary of jackknifed coefficients for each variable.
        """
        coefs = {}
        progress = tqdm(self.df.index, desc="Jackknife running...") if self.verbose else self.df.index
        first_iter = True
        for i in progress:
            jack_df = self.df.drop(i)
            jack_model = sm.OLS.from_formula(self.formula, jack_df, missing='drop').fit()
            jack_coef = jack_model.params
            if first_iter:
                coefs = {var: [] for var in jack_coef.index}
                first_iter = False
            for var in jack_coef.index:
                coefs[var].append(jack_coef[var])
        self.jackknife_distribution = coefs
        return coefs

    @staticmethod
    def compute_ahat(x):
        """
        Compute the acceleration factor 'a-hat' for the BCa interval.

        Parameters:
        - x: array-like, jackknife distribution of the coefficient.

        Returns:
        - ahat: float, acceleration factor.
        """
        x = np.array(x)
        xbar = np.mean(x)
        y = x - xbar
        num = np.sum(y ** 3)
        denom = 6 * (np.sum(y ** 2) ** (3 / 2.0))
        return num / denom

    def bca_estimate(self, beta_hat, beta_boot_dist, jackknife_dist, CI_type='BCa'):
        """
        Compute the Bias-Corrected and Accelerated (BCa) confidence interval for an estimate.
        
        Parameters:
        - beta_hat: float, original OLS estimate from the full sample.
        - beta_boot_dist: array-like, bootstrapped distribution of the estimate.
        - jackknife_dist: array-like, jackknife distribution of the estimate.
        - CI_type: str, type of confidence interval to use ('BCa', 'BC', 'perc').

        Returns:
        - beta_hat: original OLS estimate.
        - bias_corrected_beta: bias-corrected estimate.
        - ci: tuple, lower and upper bounds of the chosen confidence interval.
        """
        beta_boot_dist = np.array(beta_boot_dist)
        bias_corrected_beta = 2 * beta_hat - np.mean(beta_boot_dist)

        if CI_type == 'BCa':  # See Hansen (2020), Chapter 10.18
            ahat = self.compute_ahat(jackknife_dist)  # acceleration factor
            p_star = np.mean(beta_boot_dist < beta_hat)
            z0 = norm.ppf(p_star)
            z_low = norm.ppf(self.alpha / 2)
            z_high = norm.ppf(1 - self.alpha / 2)
            alpha_low = norm.cdf(z0 + (z_low + z0) / (1 - ahat * (z_low + z0)))
            alpha_high = norm.cdf(z0 + (z_high + z0) / (1 - ahat * (z_high + z0)))
            ci_low = np.quantile(beta_boot_dist, alpha_low)
            ci_high = np.quantile(beta_boot_dist, alpha_high)

        elif CI_type == 'BC':  # See Hansen (2020), eqs. (10.22) - (10.25)
            p_star = np.mean(beta_boot_dist < beta_hat)
            z0 = norm.ppf(p_star)
            z_low = norm.ppf(self.alpha / 2)
            z_high = norm.ppf(1 - self.alpha / 2)
            alpha_low = norm.cdf(z_low + 2 * z0)
            alpha_high = norm.cdf(z_high + 2 * z0)
            ci_low = np.quantile(beta_boot_dist, alpha_low)
            ci_high = np.quantile(beta_boot_dist, alpha_high)

        elif CI_type == 'perc': # percentile CI
            ci_low = np.quantile(beta_boot_dist, self.alpha / 2.0)
            ci_high = np.quantile(beta_boot_dist, 1 - self.alpha / 2.0)
        else:
            raise NotImplementedError(f"Confidence interval type {CI_type} not implemented.")

        return beta_hat, bias_corrected_beta, (ci_low, ci_high)
