"""
Comprehensive tests for BCaOLSpy package.
"""
import numpy as np
import pandas as pd
import pytest
from BCaOLSpy import BiasCorrectedOLS


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    z = np.random.randn(n)
    y = x - z + np.random.randn(n) * 0.1
    return pd.DataFrame({'y': y, 'x': x, 'z': z})


@pytest.fixture
def fitted_model(sample_data):
    """Return a fully fitted model with bootstrap and jackknife."""
    bc = BiasCorrectedOLS(
        sample_data, 'y', ['x', 'z'],
        verbose=False, random_state=123
    )
    bc.run_regression()
    bc.perform_bootstrap(n_sim=200)
    bc.perform_jackknife()
    return bc


class TestInputValidation:
    """Tests for input validation in __init__."""

    def test_df_not_dataframe(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            BiasCorrectedOLS("not a df", 'y', ['x'])

    def test_df_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BiasCorrectedOLS(pd.DataFrame(), 'y', ['x'])

    def test_dependent_var_not_string(self, sample_data):
        with pytest.raises(TypeError, match="must be a string"):
            BiasCorrectedOLS(sample_data, 123, ['x'])

    def test_dependent_var_not_in_columns(self, sample_data):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            BiasCorrectedOLS(sample_data, 'missing', ['x'])

    def test_independent_vars_not_list(self, sample_data):
        with pytest.raises(TypeError, match="must be a list of strings"):
            BiasCorrectedOLS(sample_data, 'y', 'x')

    def test_independent_vars_not_strings(self, sample_data):
        with pytest.raises(TypeError, match="must be a list of strings"):
            BiasCorrectedOLS(sample_data, 'y', [1, 2])

    def test_independent_vars_empty(self, sample_data):
        with pytest.raises(ValueError, match="cannot be empty"):
            BiasCorrectedOLS(sample_data, 'y', [])

    def test_independent_vars_missing_column(self, sample_data):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            BiasCorrectedOLS(sample_data, 'y', ['x', 'missing'])

    def test_alpha_not_numeric(self, sample_data):
        with pytest.raises(ValueError, match="must be a number"):
            BiasCorrectedOLS(sample_data, 'y', ['x'], alpha="0.05")

    def test_alpha_zero(self, sample_data):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BiasCorrectedOLS(sample_data, 'y', ['x'], alpha=0)

    def test_alpha_one(self, sample_data):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BiasCorrectedOLS(sample_data, 'y', ['x'], alpha=1)

    def test_invalid_cov_type(self, sample_data):
        with pytest.raises(ValueError, match="cov_type must be one of"):
            BiasCorrectedOLS(sample_data, 'y', ['x'], cov_type='invalid')

    def test_invalid_random_state(self, sample_data):
        with pytest.raises(TypeError, match="random_state must be"):
            BiasCorrectedOLS(sample_data, 'y', ['x'], random_state="invalid")

    def test_n_sim_not_int(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()
        with pytest.raises(ValueError, match="must be a positive integer"):
            bc.perform_bootstrap(n_sim=10.5)

    def test_n_sim_zero(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()
        with pytest.raises(ValueError, match="must be a positive integer"):
            bc.perform_bootstrap(n_sim=0)


class TestWorkflowEnforcement:
    """Tests for workflow enforcement checks."""

    def test_bootstrap_before_regression(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        with pytest.raises(RuntimeError, match="run_regression.*must be called"):
            bc.perform_bootstrap(n_sim=50)

    def test_jackknife_before_regression(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        with pytest.raises(RuntimeError, match="run_regression.*must be called"):
            bc.perform_jackknife()

    def test_compute_all_bca_before_regression(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        with pytest.raises(RuntimeError, match="run_regression.*must be called"):
            bc.compute_all_bca()

    def test_compute_all_bca_before_bootstrap(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()
        with pytest.raises(RuntimeError, match="perform_bootstrap.*must be called"):
            bc.compute_all_bca()

    def test_compute_all_bca_before_jackknife(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=123)
        bc.run_regression()
        bc.perform_bootstrap(n_sim=50)
        with pytest.raises(RuntimeError, match="perform_jackknife.*must be called"):
            bc.compute_all_bca()


class TestBasicFunctionality:
    """Tests for basic functionality."""

    def test_run_regression(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x', 'z'], verbose=False)
        coefs, conf, nobs = bc.run_regression()

        assert len(coefs) == 3  # Intercept, x, z
        assert 'Intercept' in coefs.index
        assert 'x' in coefs.index
        assert 'z' in coefs.index
        assert nobs == len(sample_data)

    def test_bootstrap_returns_numpy_arrays(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=123)
        bc.run_regression()
        boot = bc.perform_bootstrap(n_sim=100)

        assert isinstance(boot['x'], np.ndarray)
        assert boot['x'].shape == (100,)

    def test_jackknife_returns_numpy_arrays(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()
        jack = bc.perform_jackknife()

        assert isinstance(jack['x'], np.ndarray)
        assert jack['x'].shape == (len(sample_data),)

    def test_bca_estimate_all_types(self, fitted_model):
        for ci_type in ['BCa', 'BC', 'perc']:
            result = fitted_model.bca_estimate(
                fitted_model.coefs['x'],
                fitted_model.bootstrap_distribution['x'],
                fitted_model.jackknife_distribution['x'],
                CI_type=ci_type
            )
            assert len(result) == 3
            assert result[2][0] < result[2][1]  # ci_low < ci_high

    def test_invalid_ci_type(self, fitted_model):
        with pytest.raises(NotImplementedError, match="not implemented"):
            fitted_model.bca_estimate(
                fitted_model.coefs['x'],
                fitted_model.bootstrap_distribution['x'],
                fitted_model.jackknife_distribution['x'],
                CI_type='invalid'
            )


class TestComputeAllBca:
    """Tests for compute_all_bca convenience method."""

    def test_returns_dataframe(self, fitted_model):
        result = fitted_model.compute_all_bca()
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self, fitted_model):
        result = fitted_model.compute_all_bca()
        expected_cols = ['coef', 'bias_corrected', 'ci_low', 'ci_high']
        assert list(result.columns) == expected_cols

    def test_dataframe_index(self, fitted_model):
        result = fitted_model.compute_all_bca()
        assert 'Intercept' in result.index
        assert 'x' in result.index
        assert 'z' in result.index

    def test_all_ci_types(self, fitted_model):
        for ci_type in ['BCa', 'BC', 'perc']:
            result = fitted_model.compute_all_bca(CI_type=ci_type)
            assert isinstance(result, pd.DataFrame)


class TestReproducibility:
    """Tests for random_state reproducibility."""

    def test_same_random_state_same_results(self, sample_data):
        bc1 = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=123)
        bc1.run_regression()
        boot1 = bc1.perform_bootstrap(n_sim=50)

        bc2 = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=123)
        bc2.run_regression()
        boot2 = bc2.perform_bootstrap(n_sim=50)

        np.testing.assert_array_equal(boot1['x'], boot2['x'])

    def test_different_random_state_different_results(self, sample_data):
        bc1 = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=123)
        bc1.run_regression()
        boot1 = bc1.perform_bootstrap(n_sim=50)

        bc2 = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=456)
        bc2.run_regression()
        boot2 = bc2.perform_bootstrap(n_sim=50)

        assert not np.array_equal(boot1['x'], boot2['x'])

    def test_numpy_generator_accepted(self, sample_data):
        rng = np.random.default_rng(789)
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, random_state=rng)
        bc.run_regression()
        bc.perform_bootstrap(n_sim=50)  # Should not raise


class TestCovType:
    """Tests for cov_type parameter."""

    def test_default_cov_type(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        assert bc.cov_type == 'HC0'

    def test_all_cov_types_work(self, sample_data):
        for cov in ['HC0', 'HC1', 'HC2', 'HC3']:
            bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False, cov_type=cov)
            bc.run_regression()
            assert bc.coefs is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_degenerate_bootstrap_pstar_1(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()

        # Create artificial case where all bootstrap < beta_hat
        fake_boot = np.array([1, 2, 3, 4, 5])
        fake_jack = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="degenerate"):
            bc.bca_estimate(100, fake_boot, fake_jack)  # beta_hat >> bootstrap

    def test_degenerate_bootstrap_pstar_0(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        bc.run_regression()

        fake_boot = np.array([1, 2, 3, 4, 5])
        fake_jack = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="degenerate"):
            bc.bca_estimate(-100, fake_boot, fake_jack)  # beta_hat << bootstrap

    def test_constant_jackknife(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x'], verbose=False)
        constant_jack = np.array([5.0] * 50)
        ahat = bc.compute_ahat(constant_jack)
        assert ahat == 0.0  # Should handle gracefully

    def test_missing_data(self, sample_data):
        df_missing = sample_data.copy()
        df_missing.loc[0, 'y'] = np.nan
        df_missing.loc[5, 'x'] = np.nan

        bc = BiasCorrectedOLS(df_missing, 'y', ['x', 'z'], verbose=False, random_state=123)
        coefs, conf, nobs = bc.run_regression()

        assert nobs == len(sample_data) - 2  # Two rows dropped
        bc.perform_bootstrap(n_sim=50)
        bc.perform_jackknife()


class TestSpecialCharacters:
    """Tests for variable names with special characters."""

    def test_spaces_in_names(self):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            'my outcome': np.random.randn(n),
            'predictor one': np.random.randn(n)
        })

        bc = BiasCorrectedOLS(df, 'my outcome', ['predictor one'], verbose=False)
        bc.run_regression()
        assert bc.coefs is not None

    def test_operators_in_names(self):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            'y+1': np.random.randn(n),
            'x-value': np.random.randn(n)
        })

        bc = BiasCorrectedOLS(df, 'y+1', ['x-value'], verbose=False)
        bc.run_regression()
        assert bc.coefs is not None

    def test_parentheses_in_names(self):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            'outcome(log)': np.random.randn(n),
            'x(1)': np.random.randn(n)
        })

        bc = BiasCorrectedOLS(df, 'outcome(log)', ['x(1)'], verbose=False)
        bc.run_regression()
        assert bc.coefs is not None

    def test_normal_names_not_quoted(self, sample_data):
        bc = BiasCorrectedOLS(sample_data, 'y', ['x', 'z'], verbose=False)
        assert 'Q(' not in bc.formula  # Normal names should not be quoted
