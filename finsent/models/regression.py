import pycaret.regression as pyr
import pandas as pd
from typing import List, Optional, Union, Any, Literal, Dict

from joblib.memory import Memory
import logging
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE


class Regression(pyr.RegressionExperiment):
    """
    This class is a wrapper around the PyCaret RegressionExperiment class.
    It provides a simplified interface for training regression models using PyCaret by doing all the necessary data preprocessing.

    Refer to the PyCaret documentation for more information: https://pycaret.readthedocs.io/en/latest/api/regression.html#
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str = "Close",
        X: Union[List[str], Literal["*"]] = "*",
    ):
        """
        Constructor for the Regression class.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data.
        y : str
            The column to use as the independent variable. Default is "Close".
        X : Union[List[str], Literal["*"]]
            The columns to use as the dependent variables. Default is * which uses all feature columns.
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
            self.data = self.full_data[self.X + [self.y]]
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
            columns = self.X + [self.y]
            self.data = self.full_data[columns]
            self.dates = self.full_data.Date

        self.__shift_data()

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

    def setup(
        self,
        index: Union[bool, int, str, SEQUENCE_LIKE] = True,
        train_size: float = 0.7,
        test_data: Optional[DATAFRAME_LIKE] = None,
        ordinal_features: Optional[Dict[str, list]] = None,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        keep_features: Optional[List[str]] = None,
        preprocess: bool = True,
        create_date_columns: List[str] = ["day", "month", "year"],
        imputation_type: Optional[str] = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "mode",
        iterative_imputation_iters: int = 5,
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 25,
        encoding_method: Optional[Any] = None,
        rare_to_value: Optional[float] = None,
        rare_value: str = "rare",
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: Optional[float] = None,
        group_features: Optional[dict] = None,
        drop_groups: bool = False,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Optional[Union[int, float, str]] = None,
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: Union[str, Any] = "lightgbm",
        n_features_to_select: Union[int, float] = 0.2,
        transform_target: bool = False,
        transform_target_method: str = "yeo-johnson",
        custom_pipeline: Optional[Any] = None,
        custom_pipeline_position: int = -1,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "kfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
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
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It takes
        two mandatory parameters: ``data`` and ``target``. All the other parameters are
        optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')

        Parameters
        ----------
        index: bool, int, str or sequence, default = True
            Handle indices in the `data` dataframe.
                - If False: Reset to RangeIndex.
                - If True: Keep the provided index.
                - If int: Position of the column to use as index.
                - If str: Name of the column to use as index.
                - If sequence: Array with shape=(n_samples,) to use as index.


        train_size: float, default = 0.7
            Proportion of the dataset to be used for training and validation. Should be
            between 0.0 and 1.0.


        test_data: dataframe-like or None, default = None
            If not None, test_data is used as a hold-out set and `train_size` parameter
            is ignored. The columns of data and test_data must match.


        ordinal_features: dict, default = None
            Categorical features to be encoded ordinally. For example, a categorical
            feature with 'low', 'medium', 'high' values where low < medium < high can
            be passed as ordinal_features = {'column_name' : ['low', 'medium', 'high']}.


        numeric_features: list of str, default = None
            If the inferred data types are not correct, the numeric_features param can
            be used to define the data types. It takes a list of strings with column
            names that are numeric.


        categorical_features: list of str, default = None
            If the inferred data types are not correct, the categorical_features param
            can be used to define the data types. It takes a list of strings with column
            names that are categorical.


        date_features: list of str, default = None
            If the inferred data types are not correct, the date_features param can be
            used to overwrite the data types. It takes a list of strings with column
            names that are DateTime.


        text_features: list of str, default = None
            Column names that contain a text corpus. If None, no text features are
            selected.


        ignore_features: list of str, default = None
            ignore_features param can be used to ignore features during preprocessing
            and model training. It takes a list of strings with column names that are
            to be ignored.


        keep_features: list of str, default = None
            keep_features param can be used to always keep specific features during
            preprocessing, i.e. these features are never dropped by any kind of
            feature selection. It takes a list of strings with column names that are
            to be kept.


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split
            and custom transformations passed in ``custom_pipeline`` param. Data must be
            ready for modeling (no missing values, no dates, categorical data encoding),
            when preprocess is set to False.


        create_date_columns: list of str, default = ["day", "month", "year"]
            Columns to create from the date features. Note that created features
            with zero variance (e.g. the feature hour in a column that only contains
            dates) are ignored. Allowed values are datetime attributes from
            `pandas.Series.dt`. The datetime format of the feature is inferred
            automatically from the first non NaN value.


        imputation_type: str or None, default = 'simple'
            The type of imputation to use. Can be either 'simple' or 'iterative'.
            If None, no imputation of missing values is performed.


        numeric_imputation: int, float or str, default = 'mean'
            Imputing strategy for numerical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mean": Impute with mean of column.
                - "median": Impute with median of column.
                - "mode": Impute with most frequent value.
                - "knn": Impute using a K-Nearest Neighbors approach.
                - int or float: Impute with provided numerical value.


        categorical_imputation: str, default = 'mode'
            Imputing strategy for categorical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mode": Impute with most frequent value.
                - str: Impute with provided string.


        iterative_imputation_iters: int, default = 5
            Number of iterations. Ignored when ``imputation_type=simple``.


        numeric_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in numeric features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        categorical_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in categorical features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        text_features_method: str, default = "tf-idf"
            Method with which to embed the text features in the dataset. Choose
            between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).
            Be aware that the sparse matrix output of the transformer is converted
            internally to its full array. This can cause memory issues for large
            text embeddings.


        max_encoding_ohe: int, default = 25
            Categorical columns with `max_encoding_ohe` or less unique values are
            encoded using OneHotEncoding. If more, the `encoding_method` estimator
            is used. Note that columns with exactly two classes are always encoded
            ordinally. Set to below 0 to always use OneHotEncoding.


        encoding_method: category-encoders estimator, default = None
            A `category-encoders` estimator to encode the categorical columns
            with more than `max_encoding_ohe` unique values. If None,
            `category_encoders.target_encoder.TargetEncoder` is used.


        rare_to_value: float or None, default=None
            Minimum fraction of category occurrences in a categorical column.
            If a category is less frequent than `rare_to_value * len(X)`, it is
            replaced with the string in `rare_value`. Use this parameter to group
            rare categories before encoding the column. If None, ignores this step.


        rare_value: str, default="rare"
            Value with which to replace rare categories. Ignored when
            ``rare_to_value`` is None.


        polynomial_features: bool, default = False
            When set to True, new features are derived using existing numeric features.


        polynomial_degree: int, default = 2
            Degree of polynomial features. For example, if an input sample is two dimensional
            and of the form [a, b], the polynomial features with degree = 2 are:
            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


        low_variance_threshold: float or None, default = None
            Remove features with a training-set variance lower than the provided
            threshold. If 0, keep all features with non-zero variance, i.e. remove
            the features that have the same value in all samples. If None, skip
            this transformation step.


        group_features: dict or None, default = None
            When the dataset contains features with related characteristics,
            add new fetaures with the following statistical properties of that
            group: min, max, mean, std, median and mode. The parameter takes a
            dict with the group name as key and a list of feature names
            belonging to that group as value.


        drop_groups: bool, default=False
            Whether to drop the original features in the group. Ignored when
            ``group_features`` is None.

        remove_multicollinearity: bool, default = False
            When set to True, features with the inter-correlations higher than
            the defined threshold are removed. For each group, it removes all
            except the feature with the highest correlation to `y`.


        multicollinearity_threshold: float, default = 0.9
            Minimum absolute Pearson correlation to identify correlated
            features. The default value removes equal columns. Ignored when
            ``remove_multicollinearity`` is not True.


        bin_numeric_features: list of str, default = None
            To convert numeric features into categorical, bin_numeric_features parameter can
            be used. It takes a list of strings with column names to be discretized. It does
            so by using 'sturges' rule to determine the number of clusters and then apply
            KMeans algorithm. Original values of the feature are then replaced by the
            cluster label.


        remove_outliers: bool, default = False
            When set to True, outliers from the training data are removed using an
            Isolation Forest.


        outliers_method: str, default = "iforest"
            Method with which to remove outliers. Ignored when `remove_outliers=False`.
            Possible values are:
                - 'iforest': Uses sklearn's IsolationForest.
                - 'ee': Uses sklearn's EllipticEnvelope.
                - 'lof': Uses sklearn's LocalOutlierFactor.


        outliers_threshold: float, default = 0.05
            The percentage of outliers to be removed from the dataset. Ignored
            when ``remove_outliers=False``.


        transformation: bool, default = False
            When set to True, it applies the power transform to make data more Gaussian-like.
            Type of transformation is defined by the ``transformation_method`` parameter.


        transformation_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transformation`` is not True.


        normalize: bool, default = False
            When set to True, it transforms the features by scaling them to a given
            range. Type of scaling is defined by the ``normalize_method`` parameter.


        normalize_method: str, default = 'zscore'
            Defines the method for scaling. By default, normalize method is set to 'zscore'
            The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize``
            is not True. The other options are:

            - minmax: scales and translates each feature individually such that it is in
            the range of 0 - 1.
            - maxabs: scales and translates each feature individually such that the
            maximal absolute value of each feature will be 1.0. It does not
            shift/center the data, and thus does not destroy any sparsity.
            - robust: scales and translates each feature according to the Interquartile
            range. When the dataset contains outliers, robust scaler often gives
            better results.


        pca: bool, default = False
            When set to True, dimensionality reduction is applied to project the data into
            a lower dimensional space using the method defined in ``pca_method`` parameter.


        pca_method: str, default = 'linear'
            Method with which to apply PCA. Possible values are:
                - 'linear': Uses Singular Value  Decomposition.
                - 'kernel': Dimensionality reduction through the use of RBF kernel.
                - 'incremental': Similar to 'linear', but more efficient for large datasets.


        pca_components: int, float, str or None, default = None
            Number of components to keep. This parameter is ignored when `pca=False`.
                - If None: All components are kept.
                - If int: Absolute number of components.
                - If float: Such an amount that the variance that needs to be explained
                            is greater than the percentage specified by `n_components`.
                            Value should lie between 0 and 1 (ony for pca_method='linear').
                - If "mle": Minka’s MLE is used to guess the dimension (ony for pca_method='linear').


        feature_selection: bool, default = False
            When set to True, a subset of features is selected based on a feature
            importance score determined by ``feature_selection_estimator``.


        feature_selection_method: str, default = 'classic'
            Algorithm for feature selection. Choose from:
                - 'univariate': Uses sklearn's SelectKBest.
                - 'classic': Uses sklearn's SelectFromModel.
                - 'sequential': Uses sklearn's SequentialFeatureSelector.


        feature_selection_estimator: str or sklearn estimator, default = 'lightgbm'
            Classifier used to determine the feature importances. The
            estimator should have a `feature_importances_` or `coef_`
            attribute after fitting. If None, it uses LGBRegressor. This
            parameter is ignored when `feature_selection_method=univariate`.


        n_features_to_select: int or float, default = 0.2
            The maximum number of features to select with feature_selection. If <1,
            it's the fraction of starting features. Note that this parameter doesn't
            take features in ``ignore_features`` or ``keep_features`` into account
            when counting.


        transform_target: bool, default = False
            When set to True, target variable is transformed using the method defined in
            ``transform_target_method`` param. Target transformation is applied separately
            from feature transformations.


        transform_target_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transform_target`` is not True.

        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
            Addidiotnal custom transformers. If passed, they are applied to the
            pipeline last, after all the build-in transformers.


        custom_pipeline_position: int, default = -1
            Position of the custom pipeline in the overal preprocessing pipeline.
            The default value adds the custom pipeline last.


        data_split_shuffle: bool, default = True
            When set to False, prevents shuffling of rows during 'train_test_split'.


        data_split_stratify: bool or list, default = False
            Controls stratification during 'train_test_split'. When set to True, will
            stratify by target column. To stratify on any other columns, pass a list of
            column names. Ignored when ``data_split_shuffle`` is False.


        fold_strategy: str or sklearn CV generator object, default = 'kfold'
            Choice of cross validation strategy. Possible values are:

            * 'kfold'
            * 'groupkfold'
            * 'timeseries'
            * a custom CV generator object compatible with scikit-learn.

            For ``groupkfold``, column name must be passed in ``fold_groups`` parameter.
            Example: ``setup(fold_strategy="groupkfold", fold_groups="COLUMN_NAME")``

        fold: int, default = 10
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fold_shuffle: bool, default = False
            Controls the shuffle parameter of CV. Only applicable when ``fold_strategy``
            is 'kfold' or 'stratifiedkfold'. Ignored when ``fold_strategy`` is a custom
            object.


        fold_groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            When set to True, it will use GPU for training with algorithms that support it,
            and fall back to CPU if they are unavailable. When set to 'force', it will only
            use GPU-enabled algorithms and raise exceptions when they are unavailable. When
            False, all algorithms are trained using CPU only.

            GPU enabled algorithms:

            - Extreme Gradient Boosting, requires no further installation

            - CatBoost Classifier, requires no further installation
            (GPU is only enabled when data > 50,000 rows)

            - Light Gradient Boosting Machine, requires GPU installation
            https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

            - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
            Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15
            https://github.com/rapidsai/cuml


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb', 'comet_ml')
            corresponding to a logger to determine which experiment loggers to use.
            Setting to True will use just MLFlow.
            If ``wandb`` (Weights & Biases) or ``comet_ml``  is installed, will also log there.


        system_log: bool or str or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input
            is a string, use that as the path to the logging file. If the input
            already is a logger object, use that one instead.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is False.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server.
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is False.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is False.


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is False.


        engine: Optional[Dict[str, str]] = None
            The execution engines to use for the models in the form of a dict
            of `model_id: engine` - e.g. for Linear Regression ("lr", users can
            switch between "sklearn" and "sklearnex" by specifying
            `engine={"lr": "sklearnex"}`


        verbose: bool, default = True
            When set to False, Information grid is not printed.


        memory: str, bool or Memory, default=True
            Used to cache the fitted transformers of the pipeline.
                If False: No caching is performed.
                If True: A default temp directory is used.
                If str: Path to the caching directory.

        profile: bool, default = False
            When set to True, an interactive EDA report is displayed.


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns
        -------
            Global variables that can be changed using the ``set_config`` function.

        """
        return super().setup(
            data=self.data,
            target=self.y,
            index=index,
            train_size=train_size,
            test_data=test_data,
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
            preprocess=preprocess,
            create_date_columns=create_date_columns,
            imputation_type=imputation_type,
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            iterative_imputation_iters=iterative_imputation_iters,
            numeric_iterative_imputer=numeric_iterative_imputer,
            categorical_iterative_imputer=categorical_iterative_imputer,
            text_features_method=text_features_method,
            max_encoding_ohe=max_encoding_ohe,
            encoding_method=encoding_method,
            rare_to_value=rare_to_value,
            rare_value=rare_value,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
            low_variance_threshold=low_variance_threshold,
            group_features=group_features,
            drop_groups=drop_groups,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold=multicollinearity_threshold,
            bin_numeric_features=bin_numeric_features,
            remove_outliers=remove_outliers,
            outliers_method=outliers_method,
            outliers_threshold=outliers_threshold,
            transformation=transformation,
            transformation_method=transformation_method,
            normalize=normalize,
            normalize_method=normalize_method,
            pca=pca,
            pca_method=pca_method,
            pca_components=pca_components,
            feature_selection=feature_selection,
            feature_selection_method=feature_selection_method,
            feature_selection_estimator=feature_selection_estimator,
            n_features_to_select=n_features_to_select,
            transform_target=transform_target,
            transform_target_method=transform_target_method,
            custom_pipeline=custom_pipeline,
            custom_pipeline_position=custom_pipeline_position,
            data_split_shuffle=data_split_shuffle,
            data_split_stratify=data_split_stratify,
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            experiment_custom_tags=experiment_custom_tags,
            log_plots=log_plots,
            log_profile=log_profile,
            log_data=log_data,
            engine=engine,
            verbose=verbose,
            memory=memory,
            profile=profile,
            profile_kwargs=profile_kwargs,
        )
