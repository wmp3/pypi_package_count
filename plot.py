#!/usr/bin/env python3

'Script to make Bokeh plots.'

from datetime import timedelta
import sys

from bokeh.io import output_file
from bokeh.layouts import column
from bokeh.plotting import figure, show

import numpy as np
import pandas as pd

import forecast

TOOLS = 'pan,resize,wheel_zoom,box_zoom,save,undo,redo,reset'


def make_bokeh_base_figure(title, y_axis_label, width=800, height=350, x_axis_type='datetime',
                           x_axis_label='timestamp', tools=TOOLS):
    'Make a bokeh plot figure with common settings.'

    p = figure(width=width, height=height, x_axis_type=x_axis_type, x_axis_label=x_axis_label,
               y_axis_label=y_axis_label, title=title, tools=tools)
    p.left[0].formatter.use_scientific = False
    p.title.text_font_size = '1.5em'

    return p


def make_bokeh_forecast_plot(s_by_timestamp, predicted_timestamp, trailing_days=60,
                             pad_future_days=15, show_target=True, target=100000):
    ''''From series of actual timestamps and package counts, make bokeh plot of predicted days
        with trailing days as reference. Optionally, show reference to target.
    '''

    # filter historical data for trailing days
    s = s_by_timestamp.copy()
    cutoff = s.index[-1] - timedelta(days=int(trailing_days))
    s = s[s.index >= cutoff]
    s = s.resample('T').mean()
    most_recent_package_count = s[-1]

    # make bokeh plot
    p = make_bokeh_base_figure(title='PyPI Forecast to 100k Packages',
                               y_axis_label='Package Count')

    # extend index for actual and predicted
    future_idx_start = s.index[-1]
    future_idx_end = predicted_timestamp + timedelta(days=int(pad_future_days))
    future_idx = pd.DatetimeIndex(freq='T', start=future_idx_start, end=future_idx_end)
    common_idx = s.index.append(future_idx)

    # reindex actual series to common index
    s = s.reindex(common_idx)

    # make line to prediction
    daily_idx = pd.DatetimeIndex(freq='d', start=future_idx_start, end=predicted_timestamp)
    predictions = np.linspace(most_recent_package_count, target, num=len(daily_idx))
    s_pred = pd.Series(index=daily_idx, data=predictions)
    s_pred = s_pred.reindex(common_idx)

    # combine data into dataframe, dropping nan's
    dfc = pd.DataFrame(index=common_idx)
    dfc['actual'] = s
    dfc['forecast'] = s_pred
    dfc.columns = ['actual', 'forecast']
    dfc.dropna(how='all')

    # plot actual and forecast
    CIRCLE_SIZE = 6
    p.circle(dfc.index, dfc['actual'], legend='actual', color='grey', alpha=0.5, size=CIRCLE_SIZE)
    p.circle(dfc.index, dfc['forecast'], legend='forecast', color='orange', alpha=0.8,
             size=CIRCLE_SIZE)
    p.legend.location = 'top_left'

    # TODO: add annotation and large marker for target
    return p


def make_bokeh_historical_plot(s):
    'Make bokeh plot with historical package count data, returning bokeh plot object'

    p = make_bokeh_base_figure(y_axis_label='Package Count', title='Historical PyPI Package Count')
    p.circle(s.index, s.values, color='grey', alpha=0.5, size=4)

    p.xgrid.grid_line_color = None

    return p


def make_bokeh_daily_change_plot(df_day, window=7):
    'Make bokeh plot with daily package count change, returning bokeh plot object.'

    p = make_bokeh_base_figure(y_axis_label='Package Count Change',
                               title='Package Count Change per Day')
    p.circle(df_day.index, df_day['package_count_change'], color='grey', alpha=0.5, size=5,
             legend="Daily Package Count Change")
    p.line(df_day.index, df_day['package_count_change_moving_avg'].mean(), color='grey', alpha=0.8,
           legend="Daily Package Count Change Mean (Period)")
    p.line(df_day.index, df_day['package_count_change_moving_avg'], color='red', alpha=0.8,
           line_width=2, legend='Daily Package Count Change ({}-Day Moving Avg)'.
           format(window)
           )
    p.legend.location = 'top_left'

    return p


def make_bokeh_column_layout(*plots):
    'Make a column layout of Bokeh plots'

    return column(*plots)


def load_df_from_csv():
    in_csv_path = 'data/package_count.csv'
    try:
        df = pd.read_csv(in_csv_path,
                         sep='\t',
                         index_col=None,
                         parse_dates=['timestamp'],
                         keep_date_col=True,
                         infer_datetime_format=True,
                         warn_bad_lines=True
                         )
        return df
    except FileNotFoundError:
        msg = 'FileNotFoundError: {} file not found'.format(in_csv_path)
        return msg


def make_bokeh_plot_canvas(df=None, predicted_timestamp=None, window=7):
    'Make bokeh canvas of plots.'

    if not isinstance(df, pd.DataFrame): 
        df = load_df_from_csv()

    s = pd.Series(data=df['package_count'].values, index=df['timestamp'])
    df_day = forecast.make_daily_change_df(s, start_date=pd.Timestamp(2016, 1, 1),
                                           window=window)

    historical_plot = make_bokeh_historical_plot(s)
    package_count_change_plot = make_bokeh_daily_change_plot(df_day, window=window)

    if not predicted_timestamp:
        predicted_timestamp = pd.Timestamp(2017, 3, 3)

    forecast_plot = make_bokeh_forecast_plot(s, predicted_timestamp)
    canvas = make_bokeh_column_layout(forecast_plot, package_count_change_plot, historical_plot)

    return canvas


if __name__ == '__main__':

    canvas = make_bokeh_plot_canvas(df=None, predicted_timestamp=None, window=7)
    output_file('plot.html', title='PyPI Package Counts')
    show(canvas)
