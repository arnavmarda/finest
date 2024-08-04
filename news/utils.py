from typing import Tuple, Any, Dict


def format_date(
    start_date: Tuple[int, int, int], end_date: Tuple[int, int, int]
) -> str:
    """
    Function to format the start and end date for the URL.

    Parameters
    ----------
    start_date : Tuple[int, int, int]
        A tuple of integers representing the start date. Format: (Month, Day, Year)
    end_date : Tuple[int, int, int]
        A tuple of integers representing the end date. Format: (Month, Day, Year)

    Returns
    -------
    str
        A formatted string of the start and end date.
    """
    start_date_str = f"cd_min:{start_date[0]}/{start_date[1]}/{start_date[2]}"
    end_date_str = f"cd_max:{end_date[0]}/{end_date[1]}/{end_date[2]}"

    return f"cdr:1,{start_date_str},{end_date_str}"


def get_params(
    ticker: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    start: int,
) -> Dict["str", Any]:
    """
    Function to get the parameters for the URL.

    Parameters
    ----------
    ticker : str
        A string representing the stock ticker.
    start_date : Tuple[int, int, int]
        A tuple of integers representing the start date. Format: (Month, Day, Year)
    end_date : Tuple[int, int, int]
        A tuple of integers representing the end date. Format: (Month, Day, Year)
    start : int
        An integer representing the start index for the search.

    Returns
    -------
    Dict[str, Any]
        A dictionary of parameters for the URL.
    """
    return {
        "q": ticker,
        "tbs": format_date(start_date, end_date),
        "tbm": "nws",
        "start": start,
    }
