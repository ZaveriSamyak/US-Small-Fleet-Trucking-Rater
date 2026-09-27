"""
Legacy credibility-weighted frequency factor.

Retained as the comparison baseline for the GLM (Steps 17A-17H) and as the
fallback whenever the GLM can't score a submission. Set
FREQUENCY_METHOD = "legacy" in the notebook config to make this the live
pricing path again.
"""
import numpy as np
import pandas as pd


def calculate_frequency_factor(crashes, power_units, portfolio_frequency, credibility_weight):
    if pd.isna(power_units) or power_units <= 0:
        return np.nan, np.nan, np.nan, np.nan

    observed_frequency = crashes / power_units
    credibility = power_units / (power_units + credibility_weight)
    smoothed_frequency = (
        credibility * observed_frequency + (1 - credibility) * portfolio_frequency
    )
    raw_relativity = (
        smoothed_frequency / portfolio_frequency if portfolio_frequency > 0 else 1.0
    )

    # Prototype cap to prevent extreme factors -- needs discussion with the
    # underwriting team; currently an arbitrary 0.50x-3.00x band.
    relativity = np.clip(raw_relativity, 0.50, 3.00)

    return observed_frequency, credibility, smoothed_frequency, relativity
