import ta
import pandas as pd
import ta.volatility

# Credit to https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html
# for the implementations.

# These set of functions implement the ta library functions to make them easier to use in the context of the project.

############################################################################################################
####################################### Volatility Indicators ##############################################
############################################################################################################


def add_avg_true_range(df: pd.DataFrame, window: int = 14) -> None:
    """
    Adds the Average True Range (ATR) to the DataFrame.

    Average True Range (ATR)
    -------------------------

    The Average True Range (ATR) is a tool used in technical analysis to measure volatility. Unlike many of today's popular indicators, the ATR is not used to indicate the direction of price. Rather, it is a metric used solely to measure volatility, especially volatility caused by price gaps or limit moves.

    https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the ATR. Default is 14.

    Returns
    -------
    None
    """
    df["atr"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=window
    )


def add_bollinger_mavg(df: pd.DataFrame, window: int = 20) -> None:
    """
    Adds the Bollinger Moving Average to the DataFrame.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
     ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Moving Average. Default is 20.

    Returns
    -------
    None
    """
    df["bollinger_mavg"] = ta.volatility.bollinger_mavg(df["Close"], window=window)


def add_bollinger_uband(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Upper Band to the DataFrame.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Upper Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the upper band. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_uband"] = ta.volatility.bollinger_hband(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_bollinger_lband(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Lower Band to the DataFrame.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Lower Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the lower band. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_lband"] = ta.volatility.bollinger_lband(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_bollinger_band_width(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Channel Band Width to the DataFrame.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the upper and lower bands. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_band_width"] = ta.volatility.bollinger_wband(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_bollinger_pband(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Channel Band Width to the DataFrame.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the upper and lower bands. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_band_width"] = ta.volatility.bollinger_wband(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_bollinger_uband_indicator(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Upper Band indicator to the DataFrame.

    Returns 1, if close is higher than bollinger high band. Else, return 0.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Upper Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the upper band. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_uband_indicator"] = ta.volatility.bollinger_hband_indicator(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_bollinger_lband_indicator(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2
) -> None:
    """
    Adds the Bollinger Lower Band Indicator to the DataFrame.

    Returns 1, if close is lower than bollinger low band. Else, return 0.

    Bollinger Bands
    ---------------

    Bollinger Bands (BB) are a widely popular technical analysis instrument created by John Bollinger in the early 1980’s. Bollinger Bands consist of a band of three lines which are plotted in relation to security prices. The line in the middle is usually a Simple Moving Average (SMA) set to a period of 20 days (The type of trend line and period can be changed by the trader; however a 20 day moving average is by far the most popular). The SMA then serves as a base for the Upper and Lower Bands. The Upper and Lower Bands are used as a way to measure volatility by observing the relationship between the Bands and price. Typically the Upper and Lower Bands are set to two standard deviations away from the SMA (The Middle Line); however the number of standard deviations can also be adjusted by the trader.

    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Bollinger Lower Band. Default is 20.
    window_dev : int
        The standard deviation from the middle band to the lower band. Default is 2.

    Returns
    -------
    None
    """
    df["bollinger_lband_indicator"] = ta.volatility.bollinger_lband_indicator(
        close=df["Close"], window=window, window_dev=window_dev
    )


def add_keltner_mband(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Middle Band to the DataFrame.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_mband"] = ta.volatility.keltner_channel_mband(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_uband(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Upper Band to the DataFrame.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_uband"] = ta.volatility.keltner_channel_hband(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_lband(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Lower Band to the DataFrame.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_lband"] = ta.volatility.keltner_channel_lband(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_band_width(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Band Width to the DataFrame.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_band_width"] = ta.volatility.keltner_channel_wband(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_pband(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Percentage Band to the DataFrame.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_pband"] = ta.volatility.keltner_channel_pband(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_uband_indicator(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Upper Band Indicator to the DataFrame.

    Returns 1, if close is higher than keltner high band channel. Else, return 0.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_uband_indicator"] = ta.volatility.keltner_channel_hband_indicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_keltner_lband_indicator(
    df: pd.DataFrame, window: int = 20, ema: bool = False, window_atr: int = 10
) -> None:
    """
    Adds the Keltner Channel Lower Band Indicator to the DataFrame.

    Returns 1, if close is lower than keltner low band channel. Else, return 0.

    Keltner Channel
    ---------------

    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line as well as a Lower Envelope below the Middle Line. The Middle Line is a moving average of price over a user-defined time period. Either a simple moving average or an exponential moving average are typically used. The Upper and Lower Envelopes are set a (user-defined multiple) of a range away from the Middle Line. This can be a multiple of the daily high/low range, or more commonly a multiple of the Average True Range.

    https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Keltner Channel. Default is 20.
    ema : bool
        If True, use the Exponential Moving Average. Else, use the Simple Moving Average. Default is False.
    window_atr : int
        The window used to calculate the Average True Range. Parameter only valid, if ema is True. Default is 10.

    Returns
    -------
    None
    """
    df["keltner_lband_indicator"] = ta.volatility.keltner_channel_lband_indicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        original_version=ema,
        window_atr=window_atr,
    )


def add_ulcer_index(df: pd.DataFrame, window: int = 14) -> None:
    """
    Adds the Ulcer Index to the DataFrame.

    Ulcer Index
    -----------
    The Ulcer Index (UI) is a technical indicator that measures downside risk in terms of both the depth and duration of price declines. The index increases in value as the price moves farther away from a recent high and falls as the price rises to new highs. The indicator is usually calculated over a 14-day period, with the Ulcer Index showing the percentage drawdown a trader can expect from the high over that period.

    The greater the value of the Ulcer Index, the longer it takes for a stock to get back to the former high. Simply stated, it is designed as one measure of volatility only on the downside.

    https://www.investopedia.com/terms/u/ulcerindex.asp#:~:text=The%20Ulcer%20Index%20(UI)%20is,price%20rises%20to%20new%20highs.

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Ulcer Index. Default is 14.

    Returns
    -------
    None
    """
    df["ulcer_index"] = ta.volatility.ulcer_index(df["Close"], window=window)


def add_donchian_mband(df: pd.DataFrame, window: int = 20, offset: int = 0) -> None:
    """
    Adds the Donchian Channel Middle Band to the DataFrame.

    Donchian Channel
    ---------------

    Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

    https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Donchian Channel. Default is 20.
    offset : int
        The amount to offset the channel. Default is 0.

    Returns
    -------
    None
    """
    df["donchian_mband"] = ta.volatility.donchian_channel_mband(
        high=df["High"], low=df["Low"], close=df["Close"], window=window, offset=offset
    )


def add_donchian_uband(df: pd.DataFrame, window: int = 20, offset: int = 20) -> None:
    """
    Adds the Donchian Channel Upper Band to the DataFrame.

    Donchian Channel
    ---------------

    Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

    https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Donchian Channel. Default is 20.
    offset : int
        The amount to offset the channel. Default is 0.

    Returns
    -------
    None
    """
    df["donchian_uband"] = ta.volatility.donchian_channel_hband(
        high=df["High"], low=df["Low"], close=df["Close"], window=window, offset=offset
    )


def add_donchian_lband(df: pd.DataFrame, window: int = 20, offset: int = 20) -> None:
    """
    Adds the Donchian Channel Lower Band to the DataFrame.

    Donchian Channel
    ---------------

    Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

    https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Donchian Channel. Default is 20.
    offset : int
        The amount to offset the channel. Default is 0.

    Returns
    -------
    None
    """
    df["donchian_lband"] = ta.volatility.donchian_channel_lband(
        high=df["High"], low=df["Low"], close=df["Close"], window=window, offset=offset
    )


def add_donchian_band_width(
    df: pd.DataFrame, window: int = 20, offset: int = 20
) -> None:
    """
    Adds the Donchian Channel Band Width to the DataFrame.

    Donchian Channel
    ---------------

    Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

    https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Donchian Channel. Default is 20.
    offset : int
        The amount to offset the channel. Default is 0.

    Returns
    -------
    None
    """
    df["donchian_band_width"] = ta.volatility.donchian_channel_wband(
        high=df["High"], low=df["Low"], close=df["Close"], window=window, offset=offset
    )


def add_donchian_pband(df: pd.DataFrame, window: int = 20, offset: int = 20) -> None:
    """
    Adds the Donchian Channel Percentage Band to the DataFrame.

    Donchian Channel
    ---------------

    Donchian Channels (DC) are used in technical analysis to measure a market's volatility. It is a banded indicator, similar to Bollinger Bands %B (%B). Besides measuring a market's volatility, Donchian Channels are primarily used to identify potential breakouts or overbought/oversold conditions when price reaches either the Upper or Lower Band. These instances would indicate possible trading signals.

    https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    NOTE: The function modifies the DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stock data.
    window : int
        The window used to calculate the Donchian Channel. Default is 20.
    offset : int
        The amount to offset the channel. Default is 0.

    Returns
    -------
    None
    """
    df["donchian_pband"] = ta.volatility.donchian_channel_pband(
        high=df["High"], low=df["Low"], close=df["Close"], window=window, offset=offset
    )
