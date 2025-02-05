import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import f as snedecor_f
from sklearn.linear_model import LinearRegression


class PyEvoMotionBase():
    """
    Base class for the ``PyEvoMotion`` project.

    This class contains no data and is meant to be used as a mixin (provides utility methods for the project). It is inherited by :class:`PyEvoMotion`.
    """
    
    @staticmethod
    def count_prefixes(prefix: str, mutations: list[str]) -> int:
        """
        Count the number of mutations that start with a specific prefix.

        :param prefix: The prefix to count. It must be a single character.
        :type prefix: str
        :param mutations: The list of mutations where to count the prefix.
        :type mutations: list[str]
        :return: The number of mutations that start with the prefix.
        :rtype: ``int``
        """
        return len(list(filter(
            lambda x: x.startswith(prefix),
            mutations
        )))

    @staticmethod
    def mutation_length_modification(mutation:str) -> int:
        """
        Get the length modification induced by a mutation.

        :param mutation: The mutation whose length modification to get.
        :type mutation: str
        :return: The length modification induced by the mutation.
        :rtype: ``int``
        :raises ValueError: If the mutation is not one of ``s``, ``i`` or ``d``.
        """

        if mutation.startswith("s"): return 0
        else: _len = len(mutation.split("_")[-1])
            
        if mutation.startswith("i"): return _len
        elif mutation.startswith("d"): return -_len

        raise ValueError(f"Mutation not recognized: {mutation}")        

    @staticmethod
    def date_grouper(df: pd.DataFrame, DT: str, origin: str) -> pd.core.groupby.generic.DataFrameGroupBy:
        """
        Create grouped dataframe based on a ``datetime`` frequency.

        :param df: The dataframe to group. It must have a ``date`` column.
        :type df: pd.DataFrame
        :param DT: The string datetime that will govern the grouping.
        :type DT: str
        :param origin: The string datetime that will be the origin of the grouping frequency.
        :type origin: str
        :return grouped: The dataset's corresponding pandas groupby object.
        :rtype: ``pd.core.groupby.generic.DataFrameGroupBy``
        """

        return df.groupby(
            pd.Grouper(
                key="date",
                axis=0,
                freq=DT,
                origin=origin
            )
        )

    @staticmethod
    def _invoke_method(
        instance: object,
        method: str,
        *args: any,
        **kwargs: dict[str, any]
    ) -> any:
        """
        General method to invoke another method from a class instance.

        :param instance: the instance to invoke the method from.
        :type instance: object
        :param method: the method to invoke.
        :type method: str
        :param args: the arguments to pass to the method.
        :type args: any
        :param kwargs: the keyword arguments to pass to the method.
        :type kwargs: dict[str, any]
        :return: the result of the method.
        :rtype: any
        """

        try:
            return getattr(instance, method)(*args, **kwargs)
        except AttributeError:
            print(f"Method {method} not found in {instance}")

    @staticmethod
    def _remove_nan(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove NaN values from two pandas Series and return them as numpy arrays.

        :param x: the first pandas Series.
        :type x: pd.Series
        :param y: the second pandas Series.
        :type y: pd.Series
        :return: a tuple with the two pandas Series without NaN values.
        :rtype: tuple[np.ndarray,np.ndarray]
        """

        data = pd.DataFrame({"x": x, "y": y}).dropna()

        x = data["x"].to_numpy().reshape(-1, 1)
        y = data["y"].to_numpy().reshape(-1, 1)

        return x, y

    @classmethod
    def linear_regression(cls,
        x: np.ndarray,
        y: np.ndarray,
        fit_intercept=True
    ) -> dict[str, any]:
        """
        Perform a linear regression on a set of data.

        :param x: A numpy array of the features.
        :type x: np.ndarray
        :param y: A numpy array of the target.
        :type y: np.ndarray
        :param fit_intercept: Whether to fit the intercept. Default is ``True``.
        :type fit_intercept: bool
        :return: A dictionary containing:

            * ``model``: A ``lambda`` function that computes predictions based on the fitted model.
            * ``parameters``: A dictionary with the slope of the regression line.
            * ``expression``: A string representation of the regression equation.
            * ``r2``: The :math:`R^2` score of the regression.
        :rtype: ``dict[str, any]``
        """

        reg = LinearRegression(fit_intercept=fit_intercept).fit(x,y)

        if fit_intercept:
            model = {
                "model": lambda x: reg.coef_[0][0]*x + reg.intercept_[0],
                "parameters": {
                    "m": reg.coef_[0][0],
                    "b": reg.intercept_[0]
                },
                "expression": "mx + b"
            }

        else:
            model = {
                "model": lambda x: reg.coef_[0][0]*x,
                "parameters": {
                    "m": reg.coef_[0][0],
                },
                "expression": "mx"
            }

        model["r2"] = r2_score(y, reg.predict(x))

        return model
    
    @staticmethod
    def _power_law(
        x: np.ndarray | int | float,
        a: int | float,
        b: int | float
    ) -> np.ndarray | int | float:
        """
        Power law function.

        :param x: the input.
        :type x: np.ndarray | int | float
        :param a: the coefficient.
        :type a: int | float
        :param b: the exponent.
        :type b: int | float
        :return: the result of the power law.
        :rtype: np.ndarray | int | float
        """
        
        return a*np.power(x, b)

    @classmethod
    def power_law_fit(cls, x: np.ndarray, y: np.ndarray) -> dict[str, any]:
        """
        Perform a power law fit on a set of data.

        :param x: A numpy array of the features.
        :type x: np.ndarray
        :param y: A numpy array of the target.
        :type y: np.ndarray
        :return: A dictionary containing:

            * ``model``: A ``lambda`` function that computes predictions based on the fitted model.
            * ``parameters``: A dictionary with the parameters of the fitted power law.
            * ``expression``: A string representation of the regression equation.
            * ``r2``: The :math:`R^2` score of the regression.
        :rtype: ``dict[str, any]``
        """

        try:
            _popt, _, _, _msg, _ier = curve_fit(
                cls._power_law,
                x.T.tolist()[0], y.T.tolist()[0],
                full_output=True
            )
        except RuntimeError as e:
            _ier = 0
            _msg = str(e)

        if _ier not in range(1, 5):
            print(f"{_msg}")
            _popt = [0, 0]
        
        model = {
            "model": lambda x: _popt[0]*np.power(x, _popt[1]),
            "parameters": {
                "d": _popt[0],
                "alpha": _popt[1]
            },
            "expression": "d*x^alpha",
            "r2": r2_score(y, cls._power_law(x, *_popt))
        }

        return model
        
    @staticmethod
    def F_test(
        model1: dict[str,any],
        model2: dict[str,any],
        data: np.ndarray
    ) -> tuple[float, float]:
        """
        Perform an F-test between two models.

        See https://en.wikipedia.org/wiki/F-test#Regression_problems for more details.

        :param model1: The first model.
        :type model1: dict[str, any]
        :param model2: The second model.
        :type model2: dict[str, any]
        :param data: The data to test the models.
        :type data: np.ndarray
        :return: A tuple with the F-value and the p-value.
        :rtype: ``tuple[float, float]``
        """

        data = data.flatten()

        # Note that p1 < p2 always. Won't do an assertion because I'm making sure elsewhere that the linear model does not have an intercept, i.e. it only has the slope
        p1 = len(model1["parameters"])
        p2 = len(model2["parameters"])
        n = len(data)

        model1 = np.vectorize(model1["model"])
        model2 = np.vectorize(model2["model"])

        RS1 = (data - model1(range(n)))**2
        RS2 = (data - model2(range(n)))**2

        # Mask the infinite and nan values
        mask = (
            np.isinf(RS1)
            | np.isinf(RS2)
            | np.isnan(RS1)
            | np.isnan(RS2)
        )

        # Sum the residuals without the infinite values
        RSS1 = RS1.sum(where=~mask)
        RSS2 = RS2.sum(where=~mask)

        F = ((RSS1 - RSS2)/(p2 - p1))/(RSS2/(n - p2))

        return F, 1 - snedecor_f.cdf(F, p2 - p1, n - p2)
    
    @classmethod
    def adjust_model(cls,
        x: pd.Series,
        y: pd.Series,
        name: str = None
    ) -> dict[str, any]:
        """Adjust a model to the data.

        :param x: The features. It is a single pandas Series.
        :type x: pd.Series
        :param y: The target. It is a single pandas Series.
        :type y: pd.Series
        :param name: The name of the data. Default is ``None``.
        :type name: str
        :return: A dictionary with the model.   
        :rtype: ``dict[str, any]``
        :raises ValueError: If the dataset is empty or full of NaN values. This may occur if the grouped data contains only one entry per group, indicating that the variance cannot be computed.
        """

        x,y = cls._remove_nan(x, y)

        # Raises an error if the dataset is empty at this point
        if (x.size < 1) or (y.size < 1):
            cls._check_dataset_is_not_empty(
                pd.DataFrame(),
                f"Dataset length after filtering is: x: {x.size} elements; y: {y.size} elements. Perhaps NaN appeared certrain entries. Check if the grouped data contains only one entry per group, as this may cause NaN values when computing the variance. Also, consider widening the time window."
            )

        model1 = cls.linear_regression(x, y, fit_intercept=False) # Not fitting the intercept because data is passed scaled to the minimum
        model2 = cls.power_law_fit(x, y)

        _, p = cls.F_test(model1, model2, y)

        if p < 0.05:
            model = model2
        else:
            model = model1

        if name:
            return {name: model}
        else:
            return model

    @staticmethod
    def plot_single_data_and_model(
        data_x: pd.core.indexes.range.RangeIndex,
        data_y: pd.Series,
        data_ylabel: str,
        model: callable,
        model_label: str,
        data_xlabel_units: str,
        ax: any,
        **kwargs: dict[str, any]
    ) -> None:
        """
        Low level utility function to plot the data and a model.

        :param data_x: The x-axis data.
        :type data: pd.Series.index
        :param data_y: The y-axis data.
        :type data: pd.Series
        :param data_ylabel: The ``ylabel`` of the data.
        :type data_ylabel: str
        :param model: The model to plot.
        :type model: dict[str, any]
        :param model_label: The label of the model.
        :type model_label: str
        :param data_xlabel_units: The units of the x-axis data.
        :type data_xlabel_units: str
        :param ax: The axis to plot.
        :type ax: any
        :param kwargs: Additional arguments to pass to the plot
        :type kwargs: dict[str, any]
        """

        line_kwargs = {
            "linestyle": None,
            "color": "#1f77b4"
        }
        point_kwargs = {
            "color": "#1f77b4"
        }

        for k in kwargs.keys():
            _flag, _k = k.split("_")
            if (_k in line_kwargs) and (_flag == "line"):
                line_kwargs[_k] = kwargs[k]
            if (_k in point_kwargs) and (_flag == "point"):
                point_kwargs[_k] = kwargs[k]

        ax.scatter(
            data_x,
            data_y,
            **point_kwargs
        )
        ax.plot(
            data_x,
            model(data_x),
            label=model_label,
            **line_kwargs
        )
        ax.set_ylabel(data_ylabel)
        ax.set_xlabel(f"time ({data_xlabel_units})")
        ax.legend()

    @staticmethod
    def _check_dataset_is_not_empty(df: pd.DataFrame, msg: str) -> None:
        """Check if the dataset is not empty.

        :param df: the dataset to check.
        :type df: pd.DataFrame
        :param msg: The message to raise if the dataset is empty.
        :type msg: str
        """

        if df.empty:
            raise ValueError(
                f"The dataset is empty at this point of the analysis.\n{msg}"
            )
