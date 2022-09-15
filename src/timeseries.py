import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_adfuller_test(timeseries: pd.DataFrame, auto_lag: str = "AIC") -> bool:
    """
    Given a timeseries dataframe, this function runs Augmented Dickey-Fuller
    test to determines if timeseries is stationary or non-stationary.

    Args:
        timeseries (pd.DataFrame): dataframe with timeseries data to check if
            it is stationary or not.
        auto_lag (str): "autolag" param from statsmodels.tsa.stattools.adfuller
            function. This param refers to which method to use for automatic
            automatically determining the lag length among the values.
            Default value is "AIC".

    Notes:
        Timeseries is stationary if Augmented Dickey-Fuller test reject the
        null hypothesis (H0). Otherwise, if Augmented Dickey-Fuller test
        rejects the null hypothesis, timeseries is non-stationary.

    Returns:
        True if timeseries is stationary, otherwise False.

    """
    print('Results of Augmented Dickey-Fuller Test:')
    print('Null Hypothesis: Unit Root Present (NON-STATIONARY)')
    print('Test Statistic < Critical Value => Reject Null')
    print('P-Value =< Alpha(.05) => Reject Null\n')

    result = adfuller(timeseries.values, autolag=auto_lag)
    print(f'ADF Statistic: {result[0]}')
    p_value = result[1]
    print(f'p-value: {p_value}')

    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

    if p_value <= 0.05:
        print("Reject the null hypothesis (H0). "
              "The data is STATIONARY.")
        pass_test = True
    else:
        print("Fail to reject the null hypothesis (H0). "
              "The data is NON-STATIONARY.")
        pass_test = False

    return pass_test


def run_kpss_test(timeseries: pd.DataFrame, regression: str = 'c') -> bool:
    """
    Given a timeseries dataframe, this function runs Kwiatkowski-Phillips-
    Schmidt-Shin (KPSS) test to determines if timeseries is stationary or
    non-stationary.

    Args:
        timeseries (pd.DataFrame): dataframe with timeseries data to check if
            it is stationary or not.
        regression (str): "regression" param from
            statsmodels.tsa.stattools.kpss function.

    Notes:
        Timeseries is stationary if KPSS test reject the null hypothesis (H0).
        Otherwise, if KPSS test rejects the null hypothesis,
        timeseries is non-stationary.

    Returns:
        True if timeseries is stationary, otherwise False.

    """
    print('Results of KPSS Test:')
    print('Null Hypothesis: Data is Stationary/Trend '
          'Stationary (STATIONARY)')
    print('Test Statistic > Critical Value => Reject Null')
    print('P-Value =< Alpha(.05) => Reject Null\n')

    kpss_test = kpss(timeseries, regression=regression, nlags="auto")
    index = [
        'Test Statistic', 'p-value', 'Lags Used'
    ]
    kpss_output = pd.Series(kpss_test[0:3], index=index)
    for key, value in kpss_test[3].items():
        kpss_output[f'Critical Value {key}'] = value
    print(kpss_output)

    p_value = kpss_output[1]
    if p_value <= 0.05:
        print("Reject the null hypothesis (H0). "
              "The data is NON-STATIONARY.")
        pass_test = False
    else:
        print("Fail to reject the null hypothesis (H0). "
              "The data is STATIONARY.")
        pass_test = True

    return pass_test
