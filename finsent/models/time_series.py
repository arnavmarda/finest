import pycaret.time_series as pyts
import pandas as pd
from typing import List, Optional, Union, Any, Literal, Dict

import numpy as np
import logging
from pycaret.loggers.base_logger import BaseLogger
from sktime.forecasting.base import ForecastingHorizon


class TimeSeries(pyts.TSForecastingExperiment):

    def __init__(
        self,
        data: pd.DataFrame,
        y: str = "Close",
        X: Union[List[str], Literal["*"]] = "*",
    ):
        """
        Constructor for the TimeSeries class.

        Parameters
        -----------
        data : pd.DataFrame
            A pandas DataFrame containing the time series data.
        y : str, default = "Close"
            The name of the target variable.
        X : List[str], default = None
            A list of feature names.

        """
        super().__init__()
        self.full_data: pd.DataFrame = data

        # Check and validate independent variable
        if self.__validate_column(y):
            self.y = y
        else:
            raise ValueError(f"Target column {y} not found in the DataFrame.")

        # Check and validate dependent variables
        if X == "*":
            self.X = list(self.full_data.columns) - [
                self.y,
                "Date",
                "Open",
                "High",
                "Low",
            ]
            self.data = self.full_data[self.X + [self.y, "Date"]]
            self.dates = self.full_data.Date
        else:
            self.X = []
            for x in X:
                if self.__validate_column(x):
                    self.X.append(x)
                else:
                    raise Warning(
                        f"Dependent column {x} not found in the DataFrame. Proceeding without it."
                    )
            if self.X == []:
                raise ValueError("No valid dependent columns found in the DataFrame.")

            # Select the required columns
            columns = self.X + [self.y, "Date"]
            self.data = self.full_data[columns]
            self.dates = self.full_data.Date

        self.__shift_data()
        self.__prep_data_for_time_series()

    def __validate_column(self, column: str) -> bool:
        """
        Function to validate the column.

        Parameters
        ----------
        column : str
            The column to validate.

        Returns
        -------
        bool
            True if the column is valid, False otherwise.
        """
        if column in self.full_data.columns:
            return True
        else:
            return False

    def __shift_data(self) -> None:
        """
        Function to shift the data by one day.
        """
        for column in self.X:
            self.data[column] = self.data[column].shift(1)
        self.data = self.data.dropna()

    def __prep_data_for_time_series(self) -> None:
        """
        Function to prepare the data for time series forecasting.
        """
        self.data.Date = pd.to_datetime(self.data.Data)
        self.data.set_index("Date", inplace=True)
        self.data = self.data.asfreq("B")

    def setup(
        self,
        ignore_features: Optional[List] = None,
        numeric_imputation_target: Optional[Union[int, float, str]] = "ffill",
        numeric_imputation_exogenous: Optional[Union[int, float, str]] = "ffill",
        transform_target: Optional[str] = None,
        transform_exogenous: Optional[str] = None,
        scale_target: Optional[str] = None,
        scale_exogenous: Optional[str] = None,
        fe_target_rr: Optional[list] = None,
        fe_exogenous: Optional[list] = None,
        fold_strategy: Union[str, Any] = "expanding",
        fold: int = 3,
        fh: Optional[Union[List[int], int, np.ndarray, ForecastingHorizon]] = 1,
        hyperparameter_split: str = "all",
        seasonal_period: Optional[Union[List[Union[int, str]], int, str]] = None,
        ignore_seasonality_test: bool = False,
        sp_detection: str = "auto",
        max_sp_to_consider: Optional[int] = 60,
        remove_harmonics: bool = False,
        harmonic_order_method: str = "harmonic_max",
        num_sps_to_use: int = 1,
        seasonality_type: str = "mul",
        point_alpha: Optional[float] = None,
        coverage: Union[float, List[float]] = 0.9,
        enforce_exogenous: bool = True,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        custom_pipeline: Optional[Any] = None,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, str, logging.Logger] = True,
        log_experiment: Union[
            bool, str, BaseLogger, List[Union[str, BaseLogger]]
        ] = False,
        experiment_name: Optional[str] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        engine: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        profile: bool = False,
        profile_kwargs: Optional[Dict[str, Any]] = None,
        fig_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It takes
        one mandatory parameters: ``data``. All the other parameters are optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> airline = get_data('airline')
        >>> from pycaret.time_series import *
        >>> exp_name = setup(data = airline,  fh = 12)

        Parameters
        ----------
        ignore_features: Optional[List], default = None
            List of features to ignore for modeling when the data is a pandas
            Dataframe with more than 1 column. Ignored when data is a pandas Series
            or Dataframe with 1 column.


        numeric_imputation_target: Optional[Union[int, float, str]], default = "ffill"
            Indicates how to impute missing values in the target.
            If None, no imputation is done.
            If the target has missing values, then imputation is mandatory.
            If str, then value passed as is to the underlying `sktime` imputer.
            Allowed values are:
                "drift", "linear", "nearest", "mean", "median", "backfill",
                "bfill", "pad", "ffill", "random"
            If int or float, imputation method is set to "constant" with the given value.


        numeric_imputation_exogenous: Optional[Union[int, float, str]], default = "ffill"
            Indicates how to impute missing values in the exogenous variables.
            If None, no imputation is done.
            If exogenous variables have missing values, then imputation is mandatory.
            If str, then value passed as is to the underlying `sktime` imputer.
            Allowed values are:
                "drift", "linear", "nearest", "mean", "median", "backfill",
                "bfill", "pad", "ffill", "random"
            If int or float, imputation method is set to "constant" with the given value.


        transform_target: Optional[str], default = None
            Indicates how the target variable should be transformed.
            If None, no transformation is performed. Allowed values are
                "box-cox", "log", "sqrt", "exp", "cos"


        transform_exogenous: Optional[str], default = None
            Indicates how the exogenous variables should be transformed.
            If None, no transformation is performed. Allowed values are
                "box-cox", "log", "sqrt", "exp", "cos"


        scale_target: Optional[str], default = None
            Indicates how the target variable should be scaled.
            If None, no scaling is performed. Allowed values are
                "zscore", "minmax", "maxabs", "robust"


        scale_exogenous: Optional[str], default = None
            Indicates how the exogenous variables should be scaled.
            If None, no scaling is performed. Allowed values are
                "zscore", "minmax", "maxabs", "robust"


        fe_target_rr: Optional[list], default = None
            The transformers to be applied to the target variable in order to
            extract useful features. By default, None which means that the
            provided target variable are used "as is".

            NOTE: Most statistical and baseline models already use features (lags)
            for target variables implicitly. The only place where target features
            have to be created explicitly is in reduced regression models. Hence,
            this feature extraction is only applied to reduced regression models.

            Example::

            >>> import numpy as np
            >>> from pycaret.datasets import get_data
            >>> from sktime.transformations.series.summarize import WindowSummarizer

            >>> data = get_data("airline")

            >>> kwargs = {"lag_feature": {"lag": [36, 24, 13, 12, 11, 9, 6, 3, 2, 1]}}
            >>> fe_target_rr = [WindowSummarizer(n_jobs=1, truncate="bfill", **kwargs)]

            >>> # Baseline
            >>> exp = TSForecastingExperiment()
            >>> exp.setup(data=data, fh=12, fold=3, session_id=42)
            >>> model1 = exp.create_model("lr_cds_dt")

            >>> # With Feature Engineering
            >>> exp = TSForecastingExperiment()
            >>> exp.setup(
            >>>     data=data, fh=12, fold=3, fe_target_rr=fe_target_rr, session_id=42
            >>> )
            >>> model2 = exp.create_model("lr_cds_dt")

            >>> exp.plot_model([model1, model2], data_kwargs={"labels": ["Baseline", "With FE"]})

        fe_exogenous : Optional[list] = None
            The transformations to be applied to the exogenous variables. These
            transformations are used for all models that accept exogenous variables.
            By default, None which means that the provided exogenous variables are
            used "as is".

            Example::

            >>> import numpy as np
            >>> from sktime.transformations.series.summarize import WindowSummarizer

            >>> # Example: function num_above_thresh to count how many observations lie above
            >>> # the threshold within a window of length 2, lagged by 0 periods.
            >>> def num_above_thresh(x):
            >>>     '''Count how many observations lie above threshold.'''
            >>>     return np.sum((x > 0.7)[::-1])

            >>> kwargs1 = {"lag_feature": {"lag": [0, 1], "mean": [[0, 4]]}}
            >>> kwargs2 = {
            >>>     "lag_feature": {
            >>>         "lag": [0, 1], num_above_thresh: [[0, 2]],
            >>>         "mean": [[0, 4]], "std": [[0, 4]]
            >>>     }
            >>> }

            >>> fe_exogenous = [
            >>>     (
                        "a", WindowSummarizer(
            >>>             n_jobs=1, target_cols=["Income"], truncate="bfill", **kwargs1
            >>>         )
            >>>     ),
            >>>     (
            >>>         "b", WindowSummarizer(
            >>>             n_jobs=1, target_cols=["Unemployment", "Production"], truncate="bfill", **kwargs2
            >>>         )
            >>>     ),
            >>> ]

            >>> data = get_data("uschange")
            >>> exp = TSForecastingExperiment()
            >>> exp.setup(
            >>>     data=data, target="Consumption", fh=12,
            >>>     fe_exogenous=fe_exogenous, session_id=42
            >>> )
            >>> print(f"Feature Columns: {exp.get_config('X_transformed').columns}")
            >>> model = exp.create_model("lr_cds_dt")


        fold_strategy: str or sklearn CV generator object, default = 'expanding'
            Choice of cross validation strategy. Possible values are:

            * 'expanding'
            * 'rolling' (same as/aliased to 'expanding')
            * 'sliding'

            You can also pass an sktime compatible cross validation object such
            as ``SlidingWindowSplitter`` or ``ExpandingWindowSplitter``. In this case,
            the `fold` and `fh` parameters will be ignored and these values will
            be extracted from the ``fold_strategy`` object directly.


        fold: int, default = 3
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fh: Optional[int or list or np.array or ForecastingHorizon], default = 1
            The forecast horizon to be used for forecasting. Default is set to ``1``
            i.e. forecast one point ahead. Valid options are:
            (1) Integer: When integer is passed it means N continuous points in
                the future without any gap.
            (2) List or np.array: Indicates points to predict in the future. e.g.
                fh = [1, 2, 3, 4] or np.arange(1, 5) will predict 4 points in the future.
            (3) If you want to forecast values with gaps, you can pass an list or array
                with gaps. e.g. np.arange([13, 25]) will skip the first 12 future points
                and forecast from the 13th point till the 24th point ahead (note in numpy
                right value is inclusive and left is exclusive).
            (4) Can also be a sktime compatible ForecastingHorizon object.
            (5) If fh = None, then fold_strategy must be a sktime compatible cross validation
                object. In this case, fh is derived from this object.


        hyperparameter_split: str, default = "all"
            The split of data used to determine certain hyperparameters such as
            "seasonal_period", whether multiplicative seasonality can be used or not,
            whether the data is white noise or not, the values of non-seasonal difference
            "d" and seasonal difference "D" to use in certain models.
            Allowed values are: ["all", "train"].
            Refer for more details: https://github.com/pycaret/pycaret/issues/3202


        seasonal_period: list or int or str, default = None
            Seasonal periods to use when performing seasonality checks (i.e. candidates).

            Users can provide `seasonal_period` by passing it as an integer or a
            string corresponding to the keys below (e.g. 'W' for weekly data,
            'M' for monthly data, etc.).
                * B, C = 5
                * D = 7
                * W = 52
                * M, BM, CBM, MS, BMS, CBMS = 12
                * SM, SMS = 24
                * Q, BQ, QS, BQS = 4
                * A, Y, BA, BY, AS, YS, BAS, BYS = 1
                * H = 24
                * T, min = 60
                * S = 60

            Users can also provide a list of such values to use in models that
            accept multiple seasonal values (currently TBATS). For models that
            don't accept multiple seasonal values, the first value of the list
            will be used as the seasonal period.

            NOTE:
            (1) If seasonal_period is provided, whether the seasonality check is
            performed or not depends on the ignore_seasonality_test setting.
            (2) If seasonal_period is not provided, then the candidates are detected
            per the sp_detection setting. If seasonal_period is provided,
            sp_detection setting is ignored.


        ignore_seasonality_test: bool = False
            Whether to ignore the seasonality test or not. Applicable when seasonal_period
            is provided. If False, then a seasonality tests is performed to determine
            if the provided seasonal_period is valid or not. If it is found to be not
            valid, no seasonal period is used for modeling. If True, then the the
            provided seasonal_period is used as is.


        sp_detection: str, default = "auto"
            If seasonal_period is None, then this parameter determines the algorithm
            to use to detect the seasonal periods to use in the models.

            Allowed values are ["auto" or "index"].

            If "auto", then seasonal periods are detected using statistical tests.
            If "index", then the frequency of the data index is mapped to a seasonal
            period as shown in seasonal_period.


        max_sp_to_consider: Optional[int], default = 60,
            Max period to consider when detecting seasonal periods. If None, all
            periods up to int(("length of data"-1)/2) are considered. Length of
            the data is determined by hyperparameter_split setting.


        remove_harmonics: bool, default = False
            Should harmonics be removed when considering what seasonal periods to
            use for modeling.


        harmonic_order_method: str, default = "harmonic_max"
            Applicable when remove_harmonics = True. This determines how the harmonics
            are replaced. Allowed values are "harmonic_strength", "harmonic_max" or "raw_strength.
            - If set to  "harmonic_max", then lower seasonal period is replaced by its
            highest harmonic seasonal period in same position as the lower seasonal period.
            - If set to  "harmonic_strength", then lower seasonal period is replaced by its
            highest strength harmonic seasonal period in same position as the lower seasonal period.
            - If set to  "raw_strength", then lower seasonal periods is removed and the
            higher harmonic seasonal periods is retained in its original position
            based on its seasonal strength.

            e.g. Assuming detected seasonal periods in strength order are [2, 3, 4, 50]
            and remove_harmonics = True, then:
            - If harmonic_order_method = "harmonic_max", result = [50, 3, 4]
            - If harmonic_order_method = "harmonic_strength", result = [4, 3, 50]
            - If harmonic_order_method = "raw_strength", result = [3, 4, 50]


        num_sps_to_use: int, default = 1
            It determines the maximum number of seasonal periods to use in the models.
            Set to -1 to use all detected seasonal periods (in models that allow
            multiple seasonalities). If a model only allows one seasonal period
            and num_sps_to_use > 1, then the most dominant (primary) seasonal
            that is detected is used.


        seasonality_type : str, default = "mul"
            The type of seasonality to use. Allowed values are ["add", "mul" or "auto"]

            The detection flow sequence is as follows:
            (1) If seasonality is not detected, then seasonality type is set to None.
            (2) If seasonality is detected but data is not strictly positive, then
            seasonality type is set to "add".
            (3) If seasonality_type is "auto", then the type of seasonality is
            determined using an internal algorithm as follows
                - If seasonality is detected, then data is decomposed using
                additive and multiplicative seasonal decomposition. Then
                seasonality type is selected based on seasonality strength
                per FPP (https://otexts.com/fpp2/seasonal-strength.html). NOTE:
                For Multiplicative, the denominator multiplies the seasonal and
                residual components instead of adding them. Rest of the
                calculations remain the same. If seasonal decomposition fails for
                any reason, then defaults to multiplicative seasonality.
            (4) Otherwise, seasonality_type is set to the user provided value.


        point_alpha: Optional[float], default = None
            The alpha (quantile) value to use for the point predictions. By default
            this is set to None which uses sktime's predict() method to get the
            point prediction (the mean or the median of the forecast distribution).
            If this is set to a floating point value, then it switches to using the
            predict_quantiles() method to get the point prediction at the user
            specified quantile.
            Reference: https://robjhyndman.com/hyndsight/quantile-forecasts-in-r/

            NOTE:
            (1) Not all models support predict_quantiles(), hence, if a float
            value is provided, these models will be disabled.
            (2) Under some conditions, the user may want to only work with models
            that support prediction intervals. Utilizing note 1 to our advantage,
            the point_alpha argument can be set to 0.5 (or any float value depending
            on the quantile that the user wants to use for point predictions).
            This will disable models that do not support prediction intervals.


        coverage: Union[float, List[float]], default = 0.9
            The coverage to be used for prediction intervals (only applicable for
            models that support prediction intervals).

            If a float value is provides, it corresponds to the coverage needed
            (e.g. 0.9 means 90% coverage). This corresponds to lower and upper
            quantiles = 0.05 and 0.95 respectively.

            Alternately, if user wants to get the intervals at specific quantiles,
            a list of 2 values can be provided directly. e.g. coverage = [0.2. 0.9]
            will return the lower interval corresponding to a quantile of 0.2 and
            an upper interval corresponding to a quantile of 0.9.


        enforce_exogenous: bool, default = True
            When set to True and the data includes exogenous variables, only models
            that support exogenous variables are loaded in the environment.When
            set to False, all models are included and in this case, models that do
            not support exogenous variables will model the data as a univariate
            forecasting problem.


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            Parameter not in use for now. Behavior may change in future.


        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
            Parameter not in use for now. Behavior may change in future.


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        system_log: bool or str or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input
            is a string, use that as the path to the logging file. If the input
            already is a logger object, use that one instead.


        log_experiment: bool, default = False
            When set to True, all metrics and parameters are logged on the ``MLflow`` server.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server.
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is not True.


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is not True.


        engine: Optional[Dict[str, str]] = None
            The engine to use for the models, e.g. for auto_arima, users can
            switch between "pmdarima" and "statsforecast" by specifying
            engine={"auto_arima": "statsforecast"}


        verbose: bool, default = True
            When set to False, Information grid is not printed.


        profile: bool, default = False
            When set to True, an interactive EDA report is displayed.


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        fig_kwargs: dict, default = {} (empty dict)
            The global setting for any plots. Pass these as key-value pairs.
            Example: fig_kwargs = {"height": 1000, "template": "simple_white"}

            Available keys are:

            hoverinfo: hoverinfo passed to Plotly figures. Can be any value supported
                by Plotly (e.g. "text" to display, "skip" or "none" to disable.).
                When not provided, hovering over certain plots may be disabled by
                PyCaret when the data exceeds a  certain number of points (determined
                by `big_data_threshold`).

            renderer: The renderer used to display the plotly figure. Can be any value
                supported by Plotly (e.g. "notebook", "png", "svg", etc.). Note that certain
                renderers (like "svg") may need additional libraries to be installed. Users
                will have to do this manually since they don't come preinstalled with plotly.
                When not provided, plots use plotly's default render when data is below a
                certain number of points (determined by `big_data_threshold`) otherwise it
                switches to a static "png" renderer.

            template: The template to use for the plots. Can be any value supported by Plotly.
                If not provided, defaults to "ggplot2"

            width: The width of the plot in pixels. If not provided, defaults to None
                which lets Plotly decide the width.

            height: The height of the plot in pixels. If not provided, defaults to None
                which lets Plotly decide the height.

            rows: The number of rows to use for plots where this can be customized,
                e.g. `ccf`. If not provided, defaults to None which lets PyCaret decide
                based on number of subplots to be plotted.

            cols: The number of columns to use for plots where this can be customized,
                e.g. `ccf`. If not provided, defaults to 4

            big_data_threshold: The number of data points above which hovering over
                certain plots can be disabled and/or renderer switched to a static
                renderer. This is useful when the time series being modeled has a lot
                of data which can make notebooks slow to render. Also note that setting
                the `display_format` to a plotly-resampler figure ("plotly-dash" or
                "plotly-widget") can circumvent these problems by performing dynamic
                data aggregation.

            resampler_kwargs: The keyword arguments that are fed to configure the
                `plotly-resampler` visualizations (i.e., `display_format` "plotly-dash"
                or "plotly-widget") which down sampler will be used; how many data points
                are shown in the front-end. When the plotly-resampler figure is rendered
                via Dash (by setting the `display_format` to "plotly-dash"), one can
                also use the "show_dash" key within this dictionary to configure the
                show_dash method its args.

            Example::

                fig_kwargs = {
                    ...,
                    "resampler_kwargs":  {
                        "default_n_shown_samples": 1000,
                        "show_dash": {"mode": "inline", "port": 9012}
                    }
                }

        Returns
        -------
            Global variables that can be changed using the ``set_config`` function.
        """
        return super().setup(
            data=self.data,
            target=self.y,
            index="Date",
            ignore_features=ignore_features,
            numeric_imputation_target=numeric_imputation_target,
            numeric_imputation_exogenous=numeric_imputation_exogenous,
            transform_target=transform_target,
            transform_exogenous=transform_exogenous,
            scale_target=scale_target,
            scale_exogenous=scale_exogenous,
            fe_target_rr=fe_target_rr,
            fe_exogenous=fe_exogenous,
            fold_strategy=fold_strategy,
            fold=fold,
            fh=fh,
            hyperparameter_split=hyperparameter_split,
            seasonal_period=seasonal_period,
            ignore_seasonality_test=ignore_seasonality_test,
            sp_detection=sp_detection,
            max_sp_to_consider=max_sp_to_consider,
            remove_harmonics=remove_harmonics,
            harmonic_order_method=harmonic_order_method,
            num_sps_to_use=num_sps_to_use,
            seasonality_type=seasonality_type,
            point_alpha=point_alpha,
            coverage=coverage,
            enforce_exogenous=enforce_exogenous,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            custom_pipeline=custom_pipeline,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            experiment_custom_tags=experiment_custom_tags,
            engine=engine,
            log_plots=log_plots,
            log_profile=log_profile,
            log_data=log_data,
            verbose=verbose,
            profile=profile,
            profile_kwargs=profile_kwargs,
            fig_kwargs=fig_kwargs,
        )
