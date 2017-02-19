#!/usr/bin/env python3

# coding: utf-8

"""
Simple flask app.
"""

from flask import Flask, render_template
from datetime import datetime

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

import forecast
import plot

import pandas as pd

app = Flask(__name__)


@app.route('/')
def package_count(predicted_time=None):
    if not predicted_time:
        predicted_time = forecast.run_forecast(store_prediction=False)
    predicted_time_str = predicted_time.strftime("%A, %d %b %Y %l:%M %p")

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    df = forecast.get_data()
    fig = plot.make_bokeh_plot_canvas(df=df, predicted_timestamp=None, window=7)

    script, div = components(fig)

    html = render_template('index.html',
                                 plot_script=script,
                                 plot_div=div,
                                 js_resources=js_resources,
                                 css_resources=css_resources,
                                 predicted_time=predicted_time_str
                                 )
    return encode_utf8(html)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
