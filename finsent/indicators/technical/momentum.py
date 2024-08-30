import pandas as pd
import ta
import ta.momentum

# Credit to https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html
# for the implementations.

# These set of functions implement the ta library functions to make them easier to use in the context of the project.

############################################################################################################
######################################## Momentum Indicators ###############################################
############################################################################################################


def add_rsi(df: pd.DataFrame, window: int = 14) -> None:
    """
    Relative Strength Index (RSI)
    ----

    The Relative Strength Index (RSI) is a well versed momentum based oscillator which is used to measure
    the speed (velocity) as well as the change (magnitude) of directional price movements.
    Essentially RSI, when graphed, provides a visual mean to monitor both the current, as well as historical,
    strength and weakness of a particular market. The strength or weakness is based on closing prices over
    the duration of a specified trading period creating a reliable metric of price and momentum changes.
    Given the popularity of cash settled instruments (stock indexes) and leveraged financial products
    (the entire field of derivatives); RSI has proven to be a viable indicator of price movements.

    https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window : int
        The time period to be used in calculating the RSI. Default is 14.

    Returns
    -------
    None
    """

    df["rsi"] = ta.momentum.rsi(close=df["Close"], window=window)


def add_tsi(df: pd.DataFrame, window_slow: int = 25, window_fast: int = 13) -> None:
    """
    True Strength Index (TSI)
    ----

    The True Strength Index indicator is a momentum oscillator designed to detect, confirm or
    visualize the strength of a trend. It does this by indicating potential trends and trend
    changes through crossovers while fluctuating between positive and negative territory.
    Positive refers to buyers being in more control and negative refers to sellers being in more control.

    https://www.tradingview.com/support/solutions/43000592290-true-strength-index/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window_slow : int
        The slow time period to be used in calculating the TSI. Default is 25.
    window_fast : int
        The fast time period to be used in calculating the TSI. Default is 13.

    Returns
    -------
    None
    """
    df["tsi"] = ta.momentum.tsi(
        close=df["Close"], window_slow=window_slow, window_fast=window_fast
    )


def add_uo(
    df: pd.DataFrame,
    window1: int = 7,
    window2: int = 14,
    window3: int = 28,
    weight1: float = 4,
    weight2: float = 2,
    weight3: float = 1,
) -> None:
    """
    Ultimate Oscillator (UO)
    ----

    The Ultimate Oscillator indicator (UO) indicator is a technical analysis tool used to measure
    momentum across three varying timeframes. The problem with many momentum oscillators is that
    after a rapid advance or decline in price, they can form false divergence trading signals.
    For example, after a rapid rise in price, a bearish divergence signal may present itself,
    however price continues to rise. The ultimate Oscillator attempts to correct this by using
    multiple timeframes in its calculation as opposed to just one timeframe which is what is used
    in most other momentum oscillators.

    https://www.tradingview.com/support/solutions/43000502328-ultimate-oscillator-uo/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window1 : int
        The short time period to be used in calculating the Ultimate Oscillator. Default is 7.
    window2 : int
        The medium time period to be used in calculating the Ultimate Oscillator. Default is 14.
    window3 : int
        The long time period to be used in calculating the Ultimate Oscillator. Default is 28.
    weight1 : float
        The weight for the short BP average. Default is 4.
    weight2 : float
        The weight for the medium BP average. Default is 2.
    weight3 : float
        The weight for the long BP average. Default is 1.

    Returns
    -------
    None
    """
    df["uo"] = ta.momentum.ultimate_oscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window1=window1,
        window2=window2,
        window3=window3,
        weight1=weight1,
        weight2=weight2,
        weight3=weight3,
    )


def add_stochastic_oscillator(
    df: pd.DataFrame,
    window: int = 14,
    smooth_window: int = 3,
) -> None:
    """
    Stochastic Oscillator
    ----

    The Stochastic Oscillator (STOCH) is a range bound momentum oscillator. The Stochastic indicator
    is designed to display the location of the close compared to the high/low range over a user defined
    number of periods. Typically, the Stochastic Oscillator is used for three things; Identifying overbought
    and oversold levels, spotting divergences and also identifying bull and bear set ups or signals.

    https://www.tradingview.com/support/solutions/43000502332-stochastic-stoch/

    NOTE: Function modifies the input DataFrame in place. It adds the Stockastic Oscillator and the Stockastic Oscillator Signal.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window : int
        The time period to be used in calculating the Stochastic Oscillator. Default is 14.
    smooth_window : int
        The time period to be used in smoothing the Stochastic Oscillator. Default is 3.

    Returns
    -------
    None
    """
    df["stoch_oscillator"] = ta.momentum.stoch(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        smooth_window=smooth_window,
    )
    df["stoch_oscillator_signal"] = ta.momentum.stoch_signal(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
        smooth_window=smooth_window,
    )


def add_williams_r(df: pd.DataFrame, lbp: int = 14) -> None:
    """
    Williams %R
    ----

    Williams %R (%R) is a momentum-based oscillator used in technical analysis, primarily to identify overbought and oversold conditions. The %R is based on a comparison between the current close and the highest high for a user defined look back period. %R Oscillates between 0 and -100 (note the negative values) with readings closer to zero indicating more overbought conditions and readings closer to -100 indicating oversold. Typically %R can generate set ups based on overbought and oversold conditions as well overall changes in momentum.

    https://www.tradingview.com/support/solutions/43000502334-williams-r/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    lbp : int
        The lookback period to be used in calculating the Williams %R. Default is 14.

    Returns
    -------
    None
    """
    df["williams_r"] = ta.momentum.williams_r(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=lbp
    )


def add_awesome_oscillator(
    df: pd.DataFrame, window1: int = 5, window2: int = 34
) -> None:
    """
    Awesome Oscillator
    ----

    The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated using closing price but rather each bar's midpoints. AO is generally used to affirm trends or to anticipate possible reversals.

    https://www.tradingview.com/support/solutions/43000592288-awesome-oscillator-ao/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window1 : int
        The short time period to be used in calculating the Awesome Oscillator. Default is 5.
    window2 : int
        The long time period to be used in calculating the Awesome Oscillator. Default is 34.

    Returns
    -------
    None
    """
    df["awesome_oscillator"] = ta.momentum.ao(
        high=df["High"], low=df["Low"], window1=window1, window2=window2
    )


def add_kama(
    df: pd.DataFrame,
    window: int = 10,
    pow1: int = 2,
    pow2: int = 30,
) -> None:
    """
    Kaufman's Adaptive Moving Average (KAMA)
    ----

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window : int
        The time period to be used in calculating the KAMA. Default is 10.
    pow1 : int
        The power to be used in calculating the KAMA. Default is 2.
    pow2 : int
        The power to be used in calculating the KAMA. Default is 30.

    Returns
    -------
    None
    """
    df["kama"] = ta.momentum.kama(
        close=df["Close"], window=window, pow1=pow1, pow2=pow2
    )


def add_roc(df: pd.DataFrame, window: int = 12) -> None:
    """
    Rate of Change (ROC)
    ----

    The Rate of Change indicator (ROC) is a momentum oscillator. It calculates the percent change in price between periods. ROC takes the current price and compares it to a price "n" periods (user defined) ago. The calculated value is then plotted and fluctuates above and below a Zero Line. A technical analyst may use Rate of Change (ROC) for; trend identification, and identifying overbought and oversold conditions.

    https://www.tradingview.com/support/solutions/43000502343-rate-of-change-roc/

    NOTE: Function modifies the input DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window : int
        The time period to be used in calculating the ROC. Default is 12.

    Returns
    -------
    None
    """
    df["roc"] = ta.momentum.roc(close=df["Close"], window=window)


def add_stoch_rsi(
    df: pd.DataFrame, window: int = 14, smooth1: int = 3, smooth2: int = 3
) -> None:
    """
    Stochastic RSI
    ----

    The Stochastic RSI indicator (Stoch RSI) is essentially an indicator of an indicator. It is used in technical analysis to provide a stochastic calculation to the RSI indicator. This means that it is a measure of RSI relative to its own high/low range over a user defined period of time. The Stochastic RSI is an oscillator that calculates a value between 0 and 1 which is then plotted as a line. This indicator is primarily used for identifying overbought and oversold conditions.

    https://www.tradingview.com/support/solutions/43000502336-stochrsi-stochrsi/

    NOTE: Function modifies the input DataFrame in place. Adds the Stoch RSI, Stock RSI K, and Stock RSI D to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window : int
        The time period to be used in calculating the Stochastic RSI. Default is 14.
    smooth1 : int
        The time period to be used in smoothing the Stochastic RSI. Default is 3.
    smooth2 : int
        The time period to be used in smoothing the Stochastic RSI. Default is 3.

    Returns
    -------
    None
    """
    df["stoch_rsi"] = ta.momentum.stochrsi(
        close=df["Close"], window=window, smooth1=smooth1, smooth2=smooth2
    )
    df["stoch_rsi_k"] = ta.momentum.stochrsi_k(
        close=df["Close"], window=window, smooth1=smooth1, smooth2=smooth2
    )
    df["stoch_rsi_d"] = ta.momentum.stochrsi_d(
        close=df["Close"], window=window, smooth1=smooth1, smooth2=smooth2
    )


def add_ppo(
    df: pd.DataFrame,
    window_slow: int = 26,
    window_fast: int = 12,
    window_signal: int = 9,
) -> None:
    """
    Percentage Price Oscillator (PPO)
    ----

    The Price Oscillator indicator (PPO) is a technical analysis tool, used for measuring momentum that is very similar to the MACD. The MACD employs two Moving Averages of varying lengths (which are lagging indicators) to identify trend direction and duration. Then, MACD takes the difference in values between those two Moving Averages (MACD Line) and an EMA of those Moving Averages (Signal Line) and plots that difference between the two lines as a histogram which oscillates above and below a center Zero Line.

    PPO is exactly the same, however it then takes the same values at the MACD and calculates them as a percentage. The purpose of this, is that it makes value comparisons much more simple and straightforward over longer durations of time.

    https://www.tradingview.com/support/solutions/43000502346-price-oscillator-indicator-ppo/

    NOTE: Function modifies the input DataFrame in place. Adds the PPO, PPO Signal and PPO Hist to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window_slow : int
        The slow time period to be used in calculating the PPO. Default is 26.
    window_fast : int
        The fast time period to be used in calculating the PPO. Default is 12.
    window_signal : int
        The time period to be used in calculating the PPO Signal. Default is 9.

    Returns
    -------
    None
    """
    df["ppo"] = ta.momentum.ppo(
        close=df["Close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )
    df["ppo_signal"] = ta.momentum.ppo_signal(
        close=df["Close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )
    df["ppo_hist"] = ta.momentum.ppo_hist(
        close=df["Close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )


def add_pvo(
    df: pd.DataFrame,
    window_slow: int = 26,
    window_fast: int = 12,
    window_signal: int = 9,
) -> None:
    """
    Percentage Volume Oscillator (PVO)
    ----

    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    NOTE: Function modifies the input DataFrame in place. Adds the PVO, PVO Signal and PVO Hist to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the stock data.
    window_slow : int
        The slow time period to be used in calculating the PVO. Default is 26.
    window_fast : int
        The fast time period to be used in calculating the PVO. Default is 12.
    window_signal : int
        The time period to be used in calculating the PVO Signal. Default is 9.

    Returns
    -------
    None
    """
    df["pvo"] = ta.momentum.pvo(
        volume=df["Volume"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )
    df["pvo_signal"] = ta.momentum.pvo_signal(
        volume=df["Volume"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )
    df["pvo_hist"] = ta.momentum.pvo_hist(
        volume=df["Volume"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_signal=window_signal,
    )
