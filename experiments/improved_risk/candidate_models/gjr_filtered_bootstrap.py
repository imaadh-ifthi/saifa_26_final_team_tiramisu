"""GJR-GARCH(1,1)-filtered bootstrap simulation VaR model.

Estimates volatility with GJR-GARCH(1,1) with Student-t innovations on a trailing
window (1000 observations) refit every 21 days with warm-starts.
Standardizes historical returns by conditional volatility, resamples standardized residuals
(10,000 simulations, seeded), rescales by today's one-step-ahead GJR-GARCH forecast volatility,
and computes empirical VaR / ES from the simulated distribution.
"""
from typing import Any
import numpy as np
import pandas as pd
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from src.quantedge.tail import empirical_risk


class GJRGARCHFilteredBootstrap:
    """GJR-GARCH filtered bootstrap simulation model."""

    def __init__(self, window: int = 1000, refit_every: int = 21, n_simulations: int = 10000, seed: int = 42):
        self.window = int(window)
        self.refit_every = int(refit_every)
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)
        self.previous_params: dict[str, Any] | None = None
        self.current_params: dict[str, Any] | None = None
        self.next_refit: int = 0
        self.latest_fit_end: pd.Timestamp | None = None

    def fit_and_forecast(
        self,
        returns: pd.Series,
        evaluation_index: int,
        confidences: list[float],
        seed_offset: int = 0,
    ) -> tuple[dict[float, dict[str, Any]], dict[str, Any] | None]:
        """Fit if needed and produce one-step-ahead VaR/ES bootstrap estimates.

        Parameters
        ----------
        returns : pd.Series
            Historical log returns up to and including today.
        evaluation_index : int
            Index of the day to forecast (forecast uses returns strictly before this index).
        confidences : list[float]
            List of confidence levels.
        seed_offset : int
            Seed offset added to self.seed (typically evaluation_index) for reproducibility.

        Returns
        -------
        estimates : dict[float, dict[str, Any]]
            Mapping of confidence -> risk dict.
        diagnostic : dict[str, Any] or None
            Fit diagnostic dictionary if a refit occurred today, else None.
        """
        i = evaluation_index
        if i < self.window:
            raise ValueError(f"Need at least {self.window} prior observations before evaluation")

        sample_series = returns.iloc[i - self.window:i]
        sample = sample_series.to_numpy()
        date = returns.index[i]
        estimation_end = returns.index[i - 1]

        diagnostic = None
        if i >= self.next_refit or self.current_params is None:
            self.current_params = fit_gjr_garch(sample, self.previous_params)
            self.previous_params = self.current_params
            self.latest_fit_end = estimation_end
            self.next_refit = i + self.refit_every

            diagnostic = {
                "model": "gjr_garch",
                "fit_date": date,
                "estimation_end": self.latest_fit_end,
                "mu": self.current_params["mu"],
                "omega": self.current_params["omega"],
                "alpha": self.current_params["alpha"],
                "gamma": self.current_params["gamma"],
                "beta": self.current_params["beta"],
                "nu": self.current_params["nu"],
                "persistence": self.current_params["persistence"],
                "loglik": self.current_params["loglik"],
                "converged": self.current_params["converged"],
                "observations": self.window,
                "units": "decimal",
            }

        sigmas = filter_volatility(sample, self.current_params)
        vol = float(sigmas[-1])
        in_sample_sigmas = sigmas[:-1]

        z = (sample - self.current_params["mu"]) / in_sample_sigmas
        # Resample standardized one-step residuals, then apply the forecast
        # volatility. Keeping these steps explicit makes the bootstrap's
        # filter-and-rescale construction auditable.

        rng = np.random.default_rng(self.seed + seed_offset)
        simulated_residuals = rng.choice(z, size=self.n_simulations, replace=True)
        simulated = simulated_residuals * vol

        estimates = {}
        for c in confidences:
            risk = empirical_risk(simulated, c)
            estimates[c] = {
                **risk,
                "status": "ok" if self.current_params["converged"] else "warning_not_converged",
                "forecast_volatility": vol,
                "tail_fit_end": self.latest_fit_end,
            }

        return estimates, diagnostic
