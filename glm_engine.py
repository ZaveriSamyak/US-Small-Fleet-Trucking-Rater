"""
Poisson / Negative-Binomial GLM fitting engine.

Uses statsmodels when available. If it isn't installed, a NumPy IRLS fitter
with a statsmodels-GLMResults-compatible interface (`.params`, `.bse`,
`.pvalues`, `.llf`, `.aic`, `.deviance`, `.predict()`) takes over
transparently, so the notebook runs unmodified either way.

This is the numerical machinery, not the modelling choices -- feature
engineering, the design matrix, and which variables are fitted all live in
the notebook (Step 17B onward), where a reviewer should read them.
"""
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    sm = None
    STATSMODELS_AVAILABLE = False

try:
    from scipy.stats import norm as _NORM
    from scipy.special import gammaln as _GAMMALN
except Exception:
    _NORM = None
    _GAMMALN = np.vectorize(lambda v: __import__("math").lgamma(v))


class _IRLSResult:
    """Minimal statsmodels-GLMResults-compatible object."""

    def __init__(self, params, cov, X, y, offset, family, alpha, llf, deviance, pearson_chi2):
        self.params = pd.Series(params, index=X.columns)
        self.bse = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
        z = self.params / self.bse.replace(0, np.nan)
        self.tvalues = z
        self.pvalues = (
            pd.Series(2 * (1 - _NORM.cdf(np.abs(z.values))), index=X.columns)
            if _NORM is not None
            else pd.Series(np.nan, index=X.columns)
        )
        self.df_resid = len(y) - X.shape[1]
        self.df_model = X.shape[1] - 1
        self.llf = llf
        self.deviance = deviance
        self.pearson_chi2 = pearson_chi2
        self.aic = -2 * llf + 2 * X.shape[1]
        self.bic = -2 * llf + np.log(len(y)) * X.shape[1]
        self.family_name = family
        self.alpha = alpha
        self.nobs = len(y)

    def predict(self, X, offset=None, **kwargs):
        eta = np.asarray(X, dtype=float) @ self.params.values
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=float)
        return np.exp(np.clip(eta, -30, 30))


def _poisson_loglik(y, mu):
    return float(np.sum(y * np.log(np.maximum(mu, 1e-12)) - mu - _GAMMALN(y + 1)))


def _nb_loglik(y, mu, alpha):
    gammaln = _GAMMALN
    r = 1.0 / alpha
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(
        gammaln(y + r) - gammaln(r) - gammaln(y + 1)
        + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))
    ))


def _fit_irls(X, y, offset, family="poisson", alpha=0.0, max_iter=100, tol=1e-10):
    Xv = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    beta = np.zeros(Xv.shape[1])
    beta[0] = np.log(max(y.sum(), 1.0) / max(np.exp(offset).sum(), 1e-12))

    for _ in range(max_iter):
        eta = np.clip(Xv @ beta + offset, -30, 30)
        mu = np.exp(eta)
        # Variance: Poisson mu ; NB2 mu + alpha*mu^2
        var = mu if family == "poisson" else mu + alpha * mu ** 2
        w = (mu ** 2) / np.maximum(var, 1e-12)          # log link: (dmu/deta)^2 / var
        z = eta - offset + (y - mu) / np.maximum(mu, 1e-12)

        WX = Xv * w[:, None]
        XtWX = Xv.T @ WX
        XtWz = WX.T @ z
        new_beta = np.linalg.solve(XtWX + 1e-10 * np.eye(Xv.shape[1]), XtWz)

        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta

    eta = np.clip(Xv @ beta + offset, -30, 30)
    mu = np.exp(eta)
    var = mu if family == "poisson" else mu + alpha * mu ** 2
    w = (mu ** 2) / np.maximum(var, 1e-12)
    cov = np.linalg.inv((Xv * w[:, None]).T @ Xv + 1e-10 * np.eye(Xv.shape[1]))

    if family == "poisson":
        llf = _poisson_loglik(y, mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
        deviance = float(2 * np.sum(term - (y - mu)))
    else:
        llf = _nb_loglik(y, mu, alpha)
        r = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
            term2 = (y + r) * np.log((y + r) / (mu + r))
        deviance = float(2 * np.sum(term - term2))

    pearson = float(np.sum((y - mu) ** 2 / np.maximum(var, 1e-12)))

    return _IRLSResult(beta, cov, pd.DataFrame(Xv, columns=X.columns), y, offset,
                       family, alpha, llf, deviance, pearson)


def fit_count_glm(X, y, offset, family="poisson", alpha=None):
    """One entry point for both backends. `X` is the design matrix built by
    the notebook's make_design_matrix(); this function has no opinion about
    what's in it."""
    if STATSMODELS_AVAILABLE:
        fam = (
            sm.families.Poisson()
            if family == "poisson"
            else sm.families.NegativeBinomial(alpha=alpha)
        )
        res = sm.GLM(np.asarray(y, dtype=float), X, family=fam,
                     offset=np.asarray(offset, dtype=float)).fit()
        res.family_name = family
        res.alpha = alpha
        return res

    return _fit_irls(X, y, offset, family=family, alpha=(alpha or 0.0))
