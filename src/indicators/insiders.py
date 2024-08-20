from sec_edgar_downloader import Downloader
from tempfile import TemporaryDirectory
import glob
import html_to_json
import pandas as pd


def __get_all_files(
    directory: str, ticker: str, start_date: str, end_date: str
) -> list[str]:
    """
    Function to download all Form 4 filings for given ticker.

    Parameters
    ----------
    dir : str
        Directory to store the filings. Should use temporary directory for easy cleanup.
    ticker : str
        Ticker symbol for the company.
    start_date : str
        Start date for the filings.
    end_date : str
        End date for the filings.

    Returns
    -------
    list[str]
        List of paths to the downloaded files.
    """

    dl = Downloader("Student", "arnavmarda@gmail.com", download_folder=directory)

    if dl.get("4", ticker, after=start_date, before=end_date) > 0:
        return glob.glob(f"{directory}/sec-edgar-filings/{ticker}/*/*/*.txt")
    else:
        return []


def __parse_doc(txt: str) -> tuple[str, str, str] | None:
    """
    Function to parse the text of the filing to extract the form date, amount, and code.

    Parameters
    ----------
    text : str
        Text of the filing.

    Returns
    -------
    tuple[str, str, str]
        Tuple containing the form date, amount, and code.
    """
    json_txt = html_to_json.convert(txt)

    try:
        data = json_txt["sec-document"][0]["document"][0]["type"][0]["sequence"][0][
            "filename"
        ][0]["description"][0]["text"][0]["xml"][0]["ownershipdocument"][0][
            "derivativetable"
        ][
            0
        ][
            "derivativetransaction"
        ][
            0
        ]
    except KeyError:
        try:
            data = json_txt["sec-document"][0]["document"][0]["type"][0]["sequence"][0][
                "filename"
            ][0]["description"][0]["text"][0]["xml"][0]["ownershipdocument"][0][
                "nonderivativetable"
            ][
                0
            ][
                "nonderivativetransaction"
            ][
                0
            ]
        except KeyError:
            return None

    transaction_date = data["transactiondate"][0]["value"][0]["_value"]
    transaction_share_value = data["transactionamounts"][0]["transactionshares"][0][
        "value"
    ][0]["_value"]
    transaction_type = data["transactionamounts"][0]["transactionacquireddisposedcode"][
        0
    ]["value"][0]["_value"]

    return transaction_date, transaction_share_value, transaction_type


def __build_data_df(files: list[str]) -> pd.DataFrame:
    """
    Function to build a DataFrame from the parsed filings.

    Parameters
    ----------
    files : list[str]
        List of paths to the downloaded files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed data.
    """
    dates = []
    amount = []
    code = []
    for file in files:
        with open(file, "r") as f:
            txt = f.read()
            parsed = __parse_doc(txt)
            if parsed:
                dates.append(parsed[0])
                amount.append(int(parsed[1]))
                code.append(parsed[2])

    return pd.DataFrame({"Date": dates, "Amount": amount, "Code": code})


def __get_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Function to get the data for the given ticker and date range.

    Parameters
    ----------
    ticker : str
        Ticker symbol for the company.
    start_date : str
        Start date for the filings.
    end_date : str
        End date for the filings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed data.
    """
    with TemporaryDirectory() as directory:
        files = __get_all_files(directory, ticker, start_date, end_date)
        if len(files) == 0:
            print("Aborting Insider Trading Data Fetch. No data found.")
            return None
        df = __build_data_df(files)

    # Process Data
    df["Sign"] = df["Code"].apply(lambda x: 1 if x == "A" else -1)

    # Convert to datetime
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # Multiply amount by sign
    df["Amount"] = df["Amount"] * df["Sign"]

    df.drop(columns=["Sign", "Code"], inplace=True)

    df = df.groupby("Date").sum().reset_index()
    df.sort_values(by="Date", inplace=True)

    return df


def __merge_into_data_df(df: pd.DataFrame, insider_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to merge the insider trading data into the main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame containing the stock data.
    insider_df : pd.DataFrame
        The DataFrame containing the insider trading data.

    Returns
    -------
    pd.DataFrame
        The main DataFrame with the insider trading data merged.
    """

    # Merge the dataframes on the Date
    df = pd.merge(df, insider_df, how="left", left_on="Date", right_on="Date")

    # Fill the NaN values with 0
    df.fillna(0, inplace=True)

    return df


def add_insider_index(
    ticker: str, start_date: str, end_date: str, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to calculate and add the insider index to the existing stock data.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to calculate the insider index for.
    start_date : str
        The start date to calculate the insider index for. Format: "YYYY-MM-DD"
    end_date : str
        The end date to calculate the insider index for. Format: "YYYY-MM-DD"
    data : pd.DataFrame
        The stock data to add the insider index to.

    Returns
    -------
    pd.DataFrame
        The stock data with the insider index added.
    """
    insider_df = __get_data(ticker, start_date, end_date)

    if insider_df is None:
        return data

    data = __merge_into_data_df(data, insider_df)
    data.rename(columns={"Amount": "InsiderIndex"}, inplace=True)

    return data
