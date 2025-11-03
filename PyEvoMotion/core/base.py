import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import f as snedecor_f, t as t_dist
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
    def _remove_nan(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove NaN values from two pandas Series and return them as numpy arrays.

        :param x: the first pandas Series.
        :type x: pd.Series
        :param y: the second pandas Series.
        :type y: pd.Series
        :param z: the third pandas Series.
        :type z: pd.Series
        :return: a tuple with the two pandas Series without NaN values.
        :rtype: tuple[np.ndarray,np.ndarray]
        """

        data = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()

        x = data["x"].to_numpy().reshape(-1, 1)
        y = data["y"].to_numpy().reshape(-1, 1)
        z = data["z"].to_numpy().reshape(-1, 1)
        return x, y, z

    @staticmethod
    def _weighting_function(n: int, n_0: int = 30) -> np.ndarray:
        """
        Weighting function for the data points.

        :param n: The number of data points.
        :type n: int
        :param n_0: The number of data points at which the weighting function approximates the constant 1. Default is 30.
        :type n_0: int
        :return: The weighting function.
        :rtype: np.ndarray
        """

        return np.tanh(2*n/n_0)

    @staticmethod
    def _compute_confidence_intervals(
        parameters: dict[str, float],
        standard_errors: dict[str, float],
        degrees_of_freedom: int,
        confidence_level: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        """
        Compute confidence intervals for parameters using t-distribution.

        :param parameters: Dictionary of parameter names and their estimated values.
        :type parameters: dict[str, float]
        :param standard_errors: Dictionary of parameter names and their standard errors.
        :type standard_errors: dict[str, float]
        :param degrees_of_freedom: Degrees of freedom for the t-distribution.
        :type degrees_of_freedom: int
        :param confidence_level: Confidence level for the intervals (default 0.95 for 95% CI).
        :type confidence_level: float
        :return: Dictionary with parameter names as keys and (lower_bound, upper_bound) tuples as values.
        :rtype: dict[str, tuple[float, float]]
        """
        alpha = 1 - confidence_level
        t_val = t_dist.ppf(1 - alpha/2, degrees_of_freedom)
        
        confidence_intervals = {}
        for param_name in parameters.keys():
            param_value = parameters[param_name]
            param_se = standard_errors[param_name]
            margin_of_error = t_val * param_se
            confidence_intervals[param_name] = (
                param_value - margin_of_error,
                param_value + margin_of_error
            )
        
        return confidence_intervals

    @classmethod
    def linear_regression(cls,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        fit_intercept: bool = True,
        confidence_level: float = 0.95
    ) -> dict[str, any]:
        """
        Perform a linear regression on a set of data.

        :param x: A numpy array of the features.
        :type x: np.ndarray
        :param y: A numpy array of the target.
        :type y: np.ndarray
        :param fit_intercept: Whether to fit the intercept. Default is ``True``.
        :type fit_intercept: bool
        :param weights: Optional weights for the data points. If provided, points with higher weights will have more influence on the fit. These weights are scaled by the weighting function tanh(2*n/n_0), where n is the number of data points and n_0 is the number of data points at which the weighting function approximates the constant 1. Default is ``None``.
        :type weights: np.ndarray | None
        :param confidence_level: Confidence level for parameter confidence intervals (default 0.95 for 95% CI).
        :type confidence_level: float
        :return: A dictionary containing:

            * ``model``: A ``lambda`` function that computes predictions based on the fitted model.
            * ``parameters``: A dictionary with the slope of the regression line.
            * ``confidence_intervals``: A dictionary with confidence intervals for each parameter.
            * ``expression``: A string representation of the regression equation.
            * ``r2``: The :math:`R^2` score of the regression.
        :rtype: ``dict[str, any]``
        """

        _weights = cls._weighting_function(weights).flatten() if weights is not None else None

        reg = LinearRegression(fit_intercept=fit_intercept).fit(x, y, sample_weight=_weights)

        # Calculate confidence intervals
        n = len(x)
        _df = n - (2 if fit_intercept else 1)  # degrees of freedom
        
        # Calculate residuals and MSE
        y_pred = reg.predict(x)
        residuals = y.flatten() - y_pred.flatten()
        
        if _weights is not None:
            # Weighted MSE
            mse = np.sum(_weights * residuals**2) / (np.sum(_weights) - (2 if fit_intercept else 1))
        else:
            mse = np.sum(residuals**2) / _df
        
        # Calculate standard errors
        x_flat = x.flatten()
        x_mean = np.mean(x_flat)
        sxx = np.sum((x_flat - x_mean)**2)
        
        # Standard error for slope
        se_slope = np.sqrt(mse / sxx)

        parameters = {"m": reg.coef_[0][0]}
        standard_errors = {"m": se_slope}
        
        if fit_intercept:
            se_intercept = np.sqrt(mse * (1/n + x_mean**2/sxx))
            parameters["b"] = reg.intercept_[0]
            standard_errors["b"] = se_intercept
        
        # Compute confidence intervals using the abstracted method
        confidence_intervals = cls._compute_confidence_intervals(
            parameters, standard_errors, _df, confidence_level
        )

        if fit_intercept:
            model = {
                "model": lambda x: reg.coef_[0][0]*x + reg.intercept_[0],
                "parameters": {
                    "m": reg.coef_[0][0],
                    "b": reg.intercept_[0]
                },
                "confidence_intervals": confidence_intervals,
                "expression": "mx + b",
                "confidence_level": confidence_level
            }

        else:
            model = {
                "model": lambda x: reg.coef_[0][0]*x,
                "parameters": {
                    "m": reg.coef_[0][0],
                },
                "confidence_intervals": confidence_intervals,
                "expression": "mx",
                "confidence_level": confidence_level
            }

        model["r2"] = r2_score(y, reg.predict(x), sample_weight=_weights)

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
    def power_law_fit(cls, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, confidence_level: float = 0.95) -> dict[str, any]:
        """
        Perform a power law fit on a set of data.
        
        This method fits a power law model of the form :math:`y = d \\cdot x^{\\alpha}` to the data.
        Initial parameter estimates are obtained via linear regression on log-transformed data,
        which provides better convergence than default initialization.

        :param x: A numpy array of the features.
        :type x: np.ndarray
        :param y: A numpy array of the target.
        :type y: np.ndarray
        :param weights: Optional weights for the data points. If provided, points with higher weights will have more influence on the fit. These weights are scaled by the weighting function tanh(2*n/n_0), where n is the number of data points and n_0 is the number of data points at which the weighting function approximates the constant 1. Default is ``None``.
        :type weights: np.ndarray | None
        :param confidence_level: Confidence level for parameter confidence intervals (default 0.95 for 95% CI).
        :type confidence_level: float
        :return: A dictionary containing:

            * ``model``: A ``lambda`` function that computes predictions based on the fitted model.
            * ``parameters``: A dictionary with the parameters of the fitted power law (``d`` and ``alpha``).
            * ``confidence_intervals``: A dictionary with confidence intervals for each parameter.
            * ``expression``: A string representation of the regression equation.
            * ``r2``: The :math:`R^2` score of the regression.
            * ``confidence_level``: The confidence level used for the confidence intervals.
        :rtype: ``dict[str, any]``
        """

        _weights = cls._weighting_function(weights).flatten() if weights is not None else None

        # Provide good initial parameter guesses for power law
        # Use linear regression on log-transformed data to get initial estimates
        x_flat = x.T.tolist()[0]
        y_flat = y.T.tolist()[0]
        mask = (np.array(x_flat) > 0) & (np.array(y_flat) > 0)
        x_log = np.log(np.array(x_flat)[mask])
        y_log = np.log(np.array(y_flat)[mask])
        
        # Linear regression on log-transformed data: log(y) = log(d) + alpha*log(x)
        # This gives us initial estimates for d and alpha
        if len(x_log) > 1:
            reg = LinearRegression(fit_intercept=True).fit(x_log.reshape(-1, 1), y_log.reshape(-1, 1))
            
            p0 = [np.exp(reg.intercept_[0]), reg.coef_[0][0]]  # [d, alpha]
        else:
            p0 = [1.0, 1.0]  # Default fallback

        # Set reasonable bounds for power law parameters
        # d > 0 (coefficient must be positive)
        # alpha can be any real number, but constrain to reasonable range
        bounds = ([1e-10, -10], [np.inf, 10])  # [d_min, alpha_min], [d_max, alpha_max]

        try:
            _popt, _pcov, _, _msg, _ier = curve_fit(
                cls._power_law,
                x_flat, y_flat,
                p0=p0,
                bounds=bounds,
                sigma=1/np.sqrt(_weights) if _weights is not None else None,
                full_output=True
            )
        except RuntimeError as e:
            _ier = 0
            _msg = str(e)
            _pcov = np.array([[np.inf, 0], [0, np.inf]])
        except ValueError as e: # If the initial point breaks the algorithm, try again with a default initial point
            if "Residuals are not finite in the initial point" in str(e):
                p0 = [1.0, 1.0]
                _popt, _pcov, _, _msg, _ier = curve_fit(
                    cls._power_law,
                    x_flat, y_flat,
                    p0=p0,
                    bounds=bounds,
                    sigma=1/np.sqrt(_weights) if _weights is not None else None,
                    full_output=True
                )
            else:
                raise e

        if _ier not in range(1, 5):
            print(f"{_msg}")
            _popt = [0, 0]
            _pcov = np.array([[np.inf, 0], [0, np.inf]])
        
        # Calculate confidence intervals from covariance matrix
        n = len(x)
        df = n - 2  # degrees of freedom for 2 parameters
        
        # Standard errors from covariance matrix diagonal
        param_errors = np.sqrt(np.diag(_pcov))
        
        # Prepare parameters and standard errors for confidence interval computation
        parameters = {
            "d": _popt[0],
            "alpha": _popt[1]
        }
        standard_errors = {
            "d": param_errors[0],
            "alpha": param_errors[1]
        }
        
        # Compute confidence intervals using the abstracted method
        confidence_intervals = cls._compute_confidence_intervals(
            parameters, standard_errors, df, confidence_level
        )
        
        model = {
            "model": lambda x: _popt[0]*np.power(x, _popt[1]),
            "parameters": {
                "d": _popt[0],
                "alpha": _popt[1]
            },
            "confidence_intervals": confidence_intervals,
            "expression": "d*x^alpha",
            "confidence_level": confidence_level,
            "r2": r2_score(y, cls._power_law(x, *_popt), sample_weight=_weights)
        }

        return model
        
    @classmethod
    def F_test(
        cls,
        model1: dict[str,any],
        model2: dict[str,any],
        data: np.ndarray,
        weights: np.ndarray | None = None
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
        
        if weights is not None:
            _weights = cls._weighting_function(weights.flatten())
        else:
            _weights = np.ones(len(data))

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
        RSS1 = np.sum(_weights*RS1, where=~mask)
        RSS2 = np.sum(_weights*RS2, where=~mask)

        F = ((RSS1 - RSS2)/(p2 - p1))/(RSS2/(n - p2))

        return F, 1 - snedecor_f.cdf(F, p2 - p1, n - p2) 
    
    @classmethod
    def AIC(
        cls,
        model1: dict[str,any],
        model2: dict[str,any],
        data: np.ndarray,
        weights: np.ndarray | None = None
    ) -> tuple[float, float]:
        """
        Perform an AIC test between two models.

        Uses the small-sample corrected AIC with full constant terms:
            AICc = n*ln(2*pi) + n*ln(RSS/n) + n + 2k + [2k(k+1)]/(n-k-1)

        See https://en.wikipedia.org/wiki/Akaike_information_criterion for more details.

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
        
        if weights is not None:
            _weights = cls._weighting_function(weights.flatten())
        else:
            _weights = np.ones(len(data))

        k1 = len(model1["parameters"])
        k2 = len(model2["parameters"])
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
        RSS1 = np.sum(_weights*RS1, where=~mask)
        RSS2 = np.sum(_weights*RS2, where=~mask)

        # Handle edge case where RSS is 0 (perfect fit) to avoid log(0)
        if RSS1 == 0:
            RSS1 = 1e-10  # Small positive value to avoid log(0)
        if RSS2 == 0:
            RSS2 = 1e-10  # Small positive value to avoid log(0)

        const_term = n * (np.log(2*np.pi) + 1.0)
        denom1 = n - k1 - 1
        denom2 = n - k2 - 1

        # If denom <= 0, AICc is undefined; treat as +inf (no support)
        if denom1 <= 0:
            AICc1 = np.inf
        else:
            AICc1 = const_term + n * np.log(RSS1 / n) + 2 * k1 + (2 * k1 * (k1 + 1)) / denom1

        if denom2 <= 0:
            AICc2 = np.inf
        else:
            AICc2 = const_term + n * np.log(RSS2 / n) + 2 * k2 + (2 * k2 * (k2 + 1)) / denom2

        # ΔAIC: relative to best (lowest AIC)
        min_aicc = min(AICc1, AICc2)
        dAICc1 = AICc1 - min_aicc
        dAICc2 = AICc2 - min_aicc

        # Akaike weights
        rel1 = np.exp(-0.5 * dAICc1) if np.isfinite(dAICc1) else 0
        rel2 = np.exp(-0.5 * dAICc2) if np.isfinite(dAICc2) else 0
        denom = rel1 + rel2 if (rel1 + rel2) > 0 else 1.0
        w1 = rel1 / denom
        w2 = rel2 / denom

        return AICc1, AICc2, dAICc1, dAICc2, w1, w2 

    @classmethod
    def adjust_model(cls,
        x: pd.Series,
        y: pd.Series,
        name: str = None,
        weights: pd.Series | None = None,
        confidence_level: float = 0.95
    ) -> dict[str, any]:
        """Adjust a model to the data using AIC for model selection.

        :param x: The features. It is a single pandas Series.
        :type x: pd.Series
        :param y: The target. It is a single pandas Series.
        :type y: pd.Series
        :param name: The name of the data. Default is ``None``.
        :type name: str
        :param weights: Optional weights for the data points. If provided, points with higher weights will have more influence on the fit. These weights are scaled by the weighting function tanh(2*n/n_0), where n is the number of data points and n_0 is the number of data points at which the weighting function approximates the constant 1. Default is ``None``.
        :type weights: np.ndarray | None
        :param confidence_level: Confidence level for parameter confidence intervals (default 0.95 for 95% CI).
        :type confidence_level: float
        :return: A dictionary containing:
        
            * If name is provided: A dictionary with the name as key and the result dictionary as value
            * If name is None: A dictionary containing:
            
                * ``selected_model``: The selected model based on lowest AIC
                * ``linear_model``: The linear regression model with AIC statistics
                * ``power_law_model``: The power law model with AIC statistics
                * ``model_selection``: Dictionary with AIC comparison results
                
        :rtype: ``dict[str, any]``
        :raises ValueError: If the dataset is empty or full of NaN values. This may occur if the grouped data contains only one entry per group, indicating that the variance cannot be computed.
        """

        x,y,w = cls._remove_nan(x, y, weights)

        # Raises an error if the dataset is (almost) empty at this point
        if (x.size <= 1) or (y.size <= 1):
            cls._check_dataset_is_not_empty(
                pd.DataFrame(),
                f"Dataset length after filtering is: x: {x.size} elements; y: {y.size} elements. In particular:\n\nx: {x}\ny: {y}\n\nPerhaps NaN appeared for certain entries. Check if the grouped data contains only one entry per group, as this may cause NaN values when computing the variance. Also, consider widening the time window."
            )

        model1 = cls.linear_regression(x, y, weights=w, fit_intercept=False, confidence_level=confidence_level) # Not fitting the intercept because data is passed scaled to the minimum
        model2 = cls.power_law_fit(x, y, weights=w, confidence_level=confidence_level)

        # Compute AIC statistics for both models
        AIC1, AIC2, dAIC1, dAIC2, w1, w2 = cls.AIC(model1, model2, y, weights=w)

        # Select model with lowest AIC (highest Akaike weight)
        if AIC1 <= AIC2:
            selected_model = model1
            selected_model_name = "linear"
        else:
            selected_model = model2
            selected_model_name = "power_law"

        # Add AIC statistics to each model
        model1_with_aic = model1.copy()
        model1_with_aic.update({
            "AIC": AIC1,
            "delta_AIC": dAIC1,
            "akaike_weight": w1,
            "confidence_level": confidence_level
        })

        model2_with_aic = model2.copy()
        model2_with_aic.update({
            "AIC": AIC2,
            "delta_AIC": dAIC2,
            "akaike_weight": w2,
            "confidence_level": confidence_level
        })

        # Create comprehensive result dictionary
        result = {
            "selected_model": selected_model,
            "linear_model": model1_with_aic,
            "power_law_model": model2_with_aic,
            "model_selection": {
                "selected": selected_model_name,
                "linear_AIC": AIC1,
                "power_law_AIC": AIC2,
                "delta_AIC_linear": dAIC1,
                "delta_AIC_power_law": dAIC2,
                "akaike_weight_linear": w1,
                "akaike_weight_power_law": w2
            }
        }

        if name:
            return {name: result}
        else:
            return result

    @staticmethod
    def plot_single_data_and_model(
        data_x: pd.core.indexes.range.RangeIndex,
        data_y: pd.Series,
        data_ylabel: str,
        model: callable,
        model_label: str,
        data_xlabel_units: str,
        ax: any,
        dt_ratio: float,
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
            data_x.to_numpy()*dt_ratio,
            data_y,
            **point_kwargs
        )
        ax.plot(
            data_x.to_numpy()*dt_ratio,
            model(data_x.to_numpy()*dt_ratio),
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
                f"The dataset is (almost) empty at this point of the analysis.\n{msg}"
            )
        
    @staticmethod
    def _get_time_ratio(dt: str, reference: str = "7D") -> float:
        """Get the ratio of a time interval with respect to a reference interval.

        :param dt: Time interval string (e.g. "5D", "7D", "10D", "14D", "12H")
        :type dt: str
        :param reference: Reference time interval string. Default is "7D".
        :type reference: str
        :return: The ratio of dt to reference
        :rtype: float
        """

        return pd.Timedelta(dt) / pd.Timedelta(reference)

    @classmethod
    def _verify_dt(cls, dt: str) -> None:
        """Verify that the time window string is greater than 1 day.
        
        :param dt: Time window string (e.g. "5D", "7D", "10D", "14D")
        :type dt: str
        :raises ValueError: If the time window is not greater than 1 day
        """
        if cls._get_time_ratio(dt, "1D") <= 1:
            raise ValueError(f"Time window must be greater than 1 day. Got {dt}")
