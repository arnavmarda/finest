import pandas as pd


def normalize(X: pd.Series) -> pd.Series:
    """
    Function to normalize a given series.

    Parameters
    ----------
    X : pd.Series
        The series to normalize.

    Returns
    -------
    pd.Series
        The normalized series.
    """
    return 2 * (X - X.min()) / (X.max() - X.min()) - 1
