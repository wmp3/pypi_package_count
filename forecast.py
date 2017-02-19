#!/usr/bin/env python3

'Script to forecast pypi package data.'

import os

import pandas as pd

import models
from utils import DatabaseFunctions

def connect_db():
    'Get db connection.'

    connection_str = os.environ.get('DATABASE_URL')
    engine, metadata, con, raw_con = DatabaseFunctions.db_connect(connection_str,
                                                                  get_raw_con=False)

    return engine, metadata, con, raw_con


def get_data():
    'Get package data from sql'

    engine, metadata, con, raw_con = connect_db()
    df = pd.read_sql('select timestamp, package_count from package_count order by timestamp', con)

    return df


def drift_forecast(n, c, start=0):
    'Increment each number in sequence n by c'
    seq = []
    for x in range(n):
        if x == 0:
            n = start
        else:
            n += c
        seq.append(n)

    return seq


def make_forecast(s_actual, avg_package_count_change_per_hour, forecast_days=30):
    'Make hourly forecast.'

    hours_to_forecast = forecast_days * 24
    end_time = s_actual.index[-1]
    end_count = s_actual.iloc[-1].mean()
    forecast_idx = pd.DatetimeIndex(freq='H', start=end_time, periods=hours_to_forecast)
    forecast_package_count = drift_forecast(n=len(forecast_idx),
                                            c=avg_package_count_change_per_hour,
                                            start=end_count)
    s_forecast = pd.Series(forecast_package_count, index=forecast_idx)
    s_forecast.name = 'forecast_package_count'

    return s_forecast


def combine_series(s_actual, s_forecast):
    'Combine two series into dataframe with DatetimeIndex.'

    df_actual_forecast = pd.concat([s_actual, s_forecast], axis=1)
    df_actual_forecast.rename(columns={'package_count': 'actual_package_count'}, inplace=True)

    return df_actual_forecast


def make_daily_change_df(s, start_date=pd.Timestamp(2016, 1, 1), window=30):
    '''From series package count and Datetime index, return dataframe with Datetime index by day
    and package_count, package_count_change, package_count_change_moving_avg columns.
    '''
    s = s[s.index >= start_date]
    s = s.resample('D').mean()
    df = pd.DataFrame(s, columns=['package_count'])
    df = df.interpolate()
    df['package_count_change'] = df['package_count'] - df['package_count'].shift(1)
    df = df[df['package_count_change'] >= 0]
    df.dropna(inplace=True)
    df['package_count_change_moving_avg'] = \
        df['package_count_change'].rolling(window=window).mean()

    return df


def get_predicted_time(s_forecast, target=100000):
    'Get predicted datetime in UTC from s_forecast with DatetimeIndex.'

    first_hour_at_target = s_forecast.where(s_forecast >= target).dropna().index[0]
    predicted_time = first_hour_at_target.to_pydatetime()

    return predicted_time


def update_prediction(con, predicted_time):
    'Update predicted time in db.'

    table_name = models.Prediction.__table__
    ins = table_name.insert()
    con.execute(ins, predicted_datetime=predicted_time)


def run_forecast(store_prediction=True):
    'Run forecast returning predicted time as string.'

    engine, metadata, con, raw_con = connect_db()
    df = get_data()
    s_actual = pd.Series(df['package_count'])
    s_actual.index = df['timestamp'].values
    df_daily_change = make_daily_change_df(s_actual, start_date=pd.Timestamp(2016, 12, 1),
                                           window=7)
    avg_package_count_change_per_hour = \
        df_daily_change['package_count_change_moving_avg'].iloc[-1]/24
    s_forecast = make_forecast(s_actual, avg_package_count_change_per_hour, forecast_days=30)
    predicted_time = get_predicted_time(s_forecast, target=100000)
    if store_prediction:
        update_prediction(con, predicted_time)

    return predicted_time


if __name__ == '__main__':
    predicted_time = run_forecast(store_prediction=True)
    print('100k PyPi Packages Forecasted at {} UTC'.format(
        predicted_time.strftime('%Y-%m-%d %H:%M:%S')))
