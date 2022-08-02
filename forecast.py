"""
Multiple Time-Series Forecasting
Tutorial link:
https://medium.com/grabngoinfo/3-ways-for-multiple-time-series-forecasting-using-prophet-in-python-7a0709a117f9
"""


from prophet import Prophet


def train_and_forecast(group):
    """Train and forecast timeseries"""

    model = Prophet()

    model.fit(group)
    future = model.make_future_dataframe(periods=15)
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast["ticker"] = group["ticker"].iloc[0]

    return forecast[["ds", "ticker", "yhat", "yhat_upper", "yhat_lower"]]
