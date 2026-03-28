import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

class BiasCorrectedOLS:
    def __init__(self, df, dependent_var, independent_vars, alpha=0.05, verbose=True,
                 random_state=None, cov_type='HC0'):
        """
        Initialize the BiasCorrectedOLS object.

        Parameters:
        - df: pandas.DataFrame containing the data for the regression.
        - dependent_var: str, name of the dependent variable.
        - independent_vars: list of str, names of the independent variables.
        - alpha: float, significance level for confidence intervals (default=0.05).
        - verbose: bool, if True, display progress using tqdm for bootstrapping and jackknife methods.
        - random_state: int, numpy.random.Generator, or None. Controls randomness for reproducibility.
        - cov_type: str, type of heteroscedasticity-robust covariance ('HC0', 'HC1', 'HC2', 'HC3').
        """
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if len(df) == 0:
            raise ValueError("df cannot be empty")
        if not isinstance(dependent_var, str):
            raise TypeError("dependent_var must be a string")
        if dependent_var not in df.columns:
            raise ValueError(f"dependent_var '{dependent_var}' not found in DataFrame columns")
        if not isinstance(independent_vars, list) or not all(isinstance(v, str) for v in independent_vars):
            raise TypeError("independent_vars must be a list of strings")
        if len(independent_vars) == 0:
            raise ValueError("independent_vars cannot be empty")
        missing_vars = [v for v in independent_vars if v not in df.columns]
        if missing_vars:
            raise ValueError(f"independent_vars not found in DataFrame columns: {missing_vars}")
        if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
            raise ValueError("alpha must be a number between 0 and 1 (exclusive)")
        valid_cov_types = ('HC0', 'HC1', 'HC2', 'HC3')
        if cov_type not in valid_cov_types:
            raise ValueError(f"cov_type must be one of {valid_cov_types}")

        # Set up random number generator for reproducibility
        if random_state is None:
            self._rng = np.random.default_rng()
        elif isinstance(random_state, int):
            self._rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            raise TypeError("random_state must be None, an int, or a numpy.random.Generator")

        self.df = df
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.alpha = alpha
        self.verbose = verbose
        self.cov_type = cov_type
        # Use Q() to safely quote variable names that contain special characters
        def quote_if_needed(name):
            # Characters that require quoting in patsy formulas
            special_chars = set(' +-*/()[]{}:~^')
            if any(c in name for c in special_chars):
                return f'Q("{name}")'
            return name

        quoted_dep = quote_if_needed(dependent_var)
        quoted_indep = [quote_if_needed(v) for v in independent_vars]
        self.formula = f'{quoted_dep} ~ {" + ".join(quoted_indep)}'
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
        model = sm.OLS.from_formula(self.formula, self.df, missing='drop').fit(cov_type=self.cov_type)
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
        - bootstrap_distribution: dict mapping variable names to numpy arrays of bootstrapped coefficients.
        """
        if not isinstance(n_sim, int) or n_sim < 1:
            raise ValueError("n_sim must be a positive integer")
        if self.coefs is None:
            raise RuntimeError("run_regression() must be called before perform_bootstrap()")

        # Run one iteration to get variable names and pre-allocate arrays
        n = len(self.df)
        bootstrap_indices = self._rng.choice(n, size=n, replace=True)
        bootstrap_df = self.df.iloc[bootstrap_indices]
        model = sm.OLS.from_formula(self.formula, bootstrap_df, missing='drop').fit()
        var_names = model.params.index.tolist()
        n_vars = len(var_names)

        # Pre-allocate numpy arrays for better performance
        coefs_array = np.zeros((n_sim, n_vars))
        coefs_array[0] = model.params.values

        progress = tqdm(range(1, n_sim), desc=desc, total=n_sim, initial=1, ascii=" ▖▘▝▗▚▞█") if self.verbose else range(1, n_sim)
        for i in progress:
            bootstrap_indices = self._rng.choice(n, size=n, replace=True)
            bootstrap_df = self.df.iloc[bootstrap_indices]
            model = sm.OLS.from_formula(self.formula, bootstrap_df, missing='drop').fit()
            coefs_array[i] = model.params.values
        
        # Store as dictionary of numpy arrays for memory efficiency
        coefs = {var: coefs_array[:, j] for j, var in enumerate(var_names)}
        self.bootstrap_distribution = coefs
        return coefs

    def perform_jackknife(self):
        """
        Perform jackknife resampling to obtain the distribution of the OLS estimates.

        Returns:
        - jackknife_distribution: dict mapping variable names to numpy arrays of jackknifed coefficients.
        """
        if self.coefs is None:
            raise RuntimeError("run_regression() must be called before perform_jackknife()")

        # Get indices and run one iteration to determine variable names
        indices = self.df.index.tolist()
        n = len(indices)
        
        # First iteration to get variable names
        jack_df = self.df.drop(indices[0])
        jack_model = sm.OLS.from_formula(self.formula, jack_df, missing='drop').fit()
        var_names = jack_model.params.index.tolist()
        n_vars = len(var_names)
        
        # Pre-allocate numpy arrays for better performance
        coefs_array = np.zeros((n, n_vars))
        coefs_array[0] = jack_model.params.values
        
        progress = tqdm(range(1, n), desc="Jackknife running...", total=n, initial=1, ascii=" ▖▘▝▗▚▞█") if self.verbose else range(1, n)
        for i in progress:
            jack_df = self.df.drop(indices[i])
            jack_model = sm.OLS.from_formula(self.formula, jack_df, missing='drop').fit()
            coefs_array[i] = jack_model.params.values
        
        # Store as dictionary of numpy arrays for memory efficiency
        coefs = {var: coefs_array[:, j] for j, var in enumerate(var_names)}
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
        x = np.asarray(x)
        xbar = x.mean()
        y = x - xbar
        y2 = y * y
        num = (y * y2).sum()  # More efficient than y**3
        denom = 6.0 * (y2.sum() ** 1.5)  # More efficient than (sum ** (3/2))
        return num / denom if denom != 0 else 0.0

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
        beta_boot_dist = np.asarray(beta_boot_dist)
        bias_corrected_beta = 2 * beta_hat - beta_boot_dist.mean()

        if CI_type == 'BCa':  # See Hansen (2020), Chapter 10.18
            ahat = self.compute_ahat(jackknife_dist)  # acceleration factor
            p_star = (beta_boot_dist < beta_hat).mean()
            if p_star == 0 or p_star == 1:
                raise ValueError(
                    f"Bootstrap distribution is degenerate (p_star={p_star}). "
                    "All bootstrap estimates are on one side of beta_hat. "
                    "Try increasing n_sim or check for data issues."
                )
            z0 = norm.ppf(p_star)
            z_low = norm.ppf(self.alpha / 2)
            z_high = norm.ppf(1 - self.alpha / 2)
            denom_low = 1 - ahat * (z_low + z0)
            denom_high = 1 - ahat * (z_high + z0)
            if denom_low == 0 or denom_high == 0:
                raise ValueError(
                    f"BCa formula has division by zero (ahat={ahat:.6f}, z0={z0:.6f}). "
                    "The acceleration factor is too extreme for this data. "
                    "Consider using CI_type='BC' or 'perc' instead."
                )
            alpha_low = norm.cdf(z0 + (z_low + z0) / denom_low)
            alpha_high = norm.cdf(z0 + (z_high + z0) / denom_high)
            ci_low = np.quantile(beta_boot_dist, alpha_low)
            ci_high = np.quantile(beta_boot_dist, alpha_high)

        elif CI_type == 'BC':  # See Hansen (2020), eqs. (10.22) - (10.25)
            p_star = (beta_boot_dist < beta_hat).mean()
            if p_star == 0 or p_star == 1:
                raise ValueError(
                    f"Bootstrap distribution is degenerate (p_star={p_star}). "
                    "All bootstrap estimates are on one side of beta_hat. "
                    "Try increasing n_sim or check for data issues."
                )
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

    def compute_all_bca(self, CI_type='BCa'):
        """
        Compute BCa confidence intervals for all coefficients.

        Convenience method that computes bias-corrected estimates and confidence
        intervals for all regression coefficients using stored bootstrap and
        jackknife distributions.

        Parameters:
        - CI_type: str, type of confidence interval ('BCa', 'BC', 'perc').

        Returns:
        - results: pandas.DataFrame with columns:
            - 'coef': original OLS coefficient
            - 'bias_corrected': bias-corrected coefficient
            - 'ci_low': lower bound of confidence interval
            - 'ci_high': upper bound of confidence interval
        """
        if self.coefs is None:
            raise RuntimeError("run_regression() must be called first")
        if self.bootstrap_distribution is None:
            raise RuntimeError("perform_bootstrap() must be called first")
        if self.jackknife_distribution is None:
            raise RuntimeError("perform_jackknife() must be called first")

        results = []
        for var in self.coefs.index:
            beta_hat = self.coefs[var]
            boot_dist = self.bootstrap_distribution[var]
            jack_dist = self.jackknife_distribution[var]

            _, bias_corrected, (ci_low, ci_high) = self.bca_estimate(
                beta_hat, boot_dist, jack_dist, CI_type=CI_type
            )
            results.append({
                'coef': beta_hat,
                'bias_corrected': bias_corrected,
                'ci_low': ci_low,
                'ci_high': ci_high
            })

        return pd.DataFrame(results, index=self.coefs.index)
