import yfinance as yf
import pandas as pd


def __download_up_down_data(ticker: str) -> pd.DataFrame:
    """
    Helper function to download the upgrades and downgrades data for a stock.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to download the data for.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the upgrades and downgrades data.
    """
    return yf.Ticker(ticker).upgrades_downgrades


def __clean_data(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Helper function to clean the upgrades and downgrades data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the upgrades and downgrades data.
    start_date : str
        The start date to filter the data. Format: "YYYY-MM-DD"
    end_date : str
        The end date to filter the data. Format: "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing the upgrades and downgrades data.
    """
    # Convert the GradeDate index to a column
    data.reset_index(inplace=True)

    # Filter the data based on the start and end date
    data = data[data.GradeDate.between(start_date, end_date)]
    data["GradeDate"] = data["GradeDate"].dt.date

    # Define the mapping for the upgrades and downgrades
    transformations = {
        "Neutral": 0,
        "Overweight": 1,
        "Buy": 1,
        "Hold": 0,
        "Outperform": 1,
        "Underweight": -1,
        "Strong Buy": 2,
        "Underperform": -1,
        "Market Perform": 0,
        "Equal-Weight": 0,
        "Sector Weight": 0,
        "Sell": -1,
        "Peer Perform": 0,
        "Reduce": -1,
        "Perform": 0,
        "Long Term Buy": 1,
        "Negative": -1,
        "Positive": 1,
        "Market Outperform": 1,
        "Sector Perform": 0,
        "Equal-weight": 0,
    }

    # Map the transformations to the data
    data["FromGrade"] = data["FromGrade"].map(transformations).fillna(0)
    data["ToGrade"] = data["ToGrade"].map(transformations).fillna(0)

    # Compute the difference between the ToGrade and FromGrade
    data["Change"] = data["ToGrade"] - data["FromGrade"]

    # Calculating the Runnign Total of the Change
    data = data.groupby("GradeDate").sum().reset_index()

    # Drop the columns that are not needed
    data.drop(columns=["Action", "ToGrade", "FromGrade", "Firm"], inplace=True)

    # Sort by the GradeDate
    data.sort_values(by="GradeDate", inplace=True)

    return data


def __merge_into_data_df(df: pd.DataFrame, ud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to merge the upgrades and downgrades data into the main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame containing the stock data.
    ud_df : pd.DataFrame
        The DataFrame containing the upgrades and downgrades data.

    Returns
    -------
    pd.DataFrame
        The main DataFrame with the upgrades and downgrades data merged.
    """

    # Merge the dataframes on the Date
    df = pd.merge(df, ud_df, how="left", left_on="Date", right_on="GradeDate")

    # Drop the redundant columns
    df.drop(columns=["GradeDate"], inplace=True)

    # Fill the NaN values with 0
    df.fillna(0, inplace=True)
    df["AnalystIndex"] = df["Change"].cumsum()
    df.drop(columns=["Change"], inplace=True)

    return df
