import typing as t

import numpy as np
from dash import Dash, html, dcc
from plotly import graph_objects as go


def scatter_plot_with_regression_line(
    x: np.ndarray,
    y: np.ndarray,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
) -> None:
    """
    Scatter plot with regression line.
    :param x: x values
    :param y: y values
    :param title: title of the plot
    :param x_label: x label
    :param y_label: y label
    :return: None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.poly1d(np.polyfit(x, y, 1))(x),
            mode="lines",
            name="Regression line",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()
