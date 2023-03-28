import typing as t

import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash as Dash
from plotly import graph_objects as go


class Model(t.Protocol):
    """A callable that takes features and parameters and returns predictions."""

    def __call__(self, x: np.ndarray, *parameters: float) -> np.ndarray:
        ...


class LossFn(t.Protocol):
    """A callable that takes predictions and labels and returns a loss."""

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        ...


class ParameterRange(t.NamedTuple):
    """A range of values for a parameter."""

    minimum: float
    maximum: float
    step: float = 0.5

    @property
    def avg(self) -> float:
        return 0.5 * (self.minimum + self.maximum)


class JaxLossFn(t.Protocol):
    """Similar to `LossFn`, but for JAX."""

    def __call__(
        self,
        params: tuple[jnp.ndarray | float, ...],
        y_pred: jnp.ndarray,
        y_true: jnp.ndarray,
    ) -> float:
        ...


def visualize_parameter_changes_app(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    parameter_ranges: dict[str, ParameterRange],
    loss_fn: LossFn,
) -> Dash:
    """Returns a Dash app for tweaking model parameters.

    The app does the following things:
    1. Display a scatter plot of the data.
    2. Display the line corresponding to the current parameter values.
    3. Display the loss corresponding to the current parameter values.
    4. Display a slider for each parameter.
    """
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(
                        id="data-plot",
                        figure=go.Figure(
                            data=[
                                go.Scatter(x=x, y=y, mode="markers", name="Data"),
                                go.Scatter(
                                    x=x,
                                    y=model(
                                        x,
                                        *(
                                            parameter_range.avg
                                            for parameter_range in parameter_ranges.values()
                                        ),
                                    ),
                                    mode="lines",
                                    name="Model",
                                ),
                            ],
                            layout=go.Layout(
                                title="Data",
                                xaxis_title="x",
                                yaxis_title="y",
                            ),
                        ),
                    ),
                    html.H2(id="loss"),
                ],
                style={"width": "100%", "display": "block"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(parameter_name),
                            dcc.Slider(
                                id=parameter_name,
                                min=parameter_range.minimum,
                                max=parameter_range.maximum,
                                step=parameter_range.step,
                                value=parameter_range.avg,
                            ),
                        ],
                        style={"width": "100%", "display": "block"},
                    )
                    for parameter_name, parameter_range in parameter_ranges.items()
                ],
                style={"width": "100%", "display": "block"},
            ),
        ]
    )

    @app.callback(
        [
            Output("data-plot", "figure"),
            Output("loss", "children"),
        ],
        [Input(parameter_name, "value") for parameter_name in parameter_ranges.keys()],
    )
    def update_plot(*parameters: float):
        fig = go.Figure(
            data=[
                go.Scatter(x=x, y=y, mode="markers", name="Data"),
                go.Scatter(
                    x=x,
                    y=model(x, *parameters),
                    mode="lines",
                    name="Model",
                ),
            ],
            layout=go.Layout(
                title="Data",
                xaxis_title="x",
                yaxis_title="y",
            ),
        )
        loss = loss_fn(model(x, *parameters), y)
        return fig, f"Loss: {loss:.2f}"

    return app


# TODO: Replace jax with pytorch
# def visualize_lr_affect_gd_app(
#     loss_fn: JaxLossFn,
#     x: None,
#     y: jnp.ndarray,
#     lr: ParameterRange,
# ) -> Dash:
#     """Returns a Dash app for visualizing the effect of the learning rate.

#     The app does the following things:
#     1. Display a plot of the loss as a function of the learning rate.
#     2. Display a slider for the learning rate.
#     """
#     grad_fn = jax.grad(loss_fn)

#     def train_step(
#         weight: float,
#         lr: float,
#     ) -> tuple[float, float]:
#         """Performs a single training step.

#         Args:
#             weight: Slope of the line.
#             lr: Learning rate.

#         Returns:
#             Loss.
#         """
#         grad = grad_fn((weight,), x, y)
#         return loss_fn((weight,), x, y), weight - lr * grad

#     def train_losses(init_weight: float, lr: float) -> tuple[list[float], list[float]]:
#         """Performs the training until the loss plateaus and returns the losses.

#         Args:
#             init_weight: Initial slope of the line.
#             lr: float

#         Returns:
#             Weight for each epoch.
#             Loss for each epoch.
#         """
#         losses = []
#         weights = [init_weight]
#         weight = init_weight
#         for _ in range(1000):
#             loss, weight = train_step(weight, lr)
#             losses.append(loss)
#             weights.append(weight)
#             if len(losses) > 100 and np.abs(losses[-1] - losses[-2]) < 1e-3:
#                 break
#         return weights, losses

#     def predict(weight: float, x: jnp.ndarray) -> jnp.ndarray:
#         """Predicts the y values for the given x values."""
#         return weight * x

#     def plot_training(lr: float) -> go.Figure:
#         """Plots a curve showing the loss_fn and the losses at each epoch."""
#         init_params = jnp.random.normal(jax.random.PRNGKey(0), ())
#         weights, losses = train_losses(init_params, lr)
#         loss_x = jnp.linspace(-1000, 1000, 1000)
#         loss_y = np.array(
#             [
#                 loss_fn((weight,), y_pred=predict(weight, x), y_true=y)
#                 for weight in loss_x
#             ]
#         )

#         fig = go.Figure(
#             data=[
#                 go.Scatter(
#                     x=loss_x,
#                     y=loss_y,
#                     mode="lines",
#                     name="Loss",
#                 ),
#                 go.Scatter(
#                     x=np.array(weights),
#                     y=np.array(losses),
#                     mode="lines",
#                     name="Training Loss",
#                 ),
#             ],
#             layout=go.Layout(
#                 title="Loss",
#                 xaxis_title="Learning Rate",
#                 yaxis_title="Loss",
#             ),
#         )
#         return fig

#     app = Dash(__name__)
#     plot = html.Div(
#         [
#             dcc.Graph(
#                 id="loss-plot",
#                 figure=plot_training(lr.avg),
#             ),
#             html.H2(id="lr"),
#         ],
#         style={"width": "100%", "display": "block"},
#     )
#     text_input = html.Div(
#         [
#             html.Label("Learning rate"),
#             html.Div(
#                 [
#                     html.H2("Learning rate"),
#                     dcc.Input(
#                         id="Learning rate",
#                         type="text",
#                         value=lr.avg,
#                     ),
#                 ],
#                 style={"width": "100%", "display": "block"},
#             ),
#         ],
#         style={"width": "100%", "display": "block"},
#     )
#     app.layout = html.Div(
#         [
#             plot,
#             html.Div(
#                 children=[text_input],
#                 stype={
#                     "display": "flex",
#                     "justifyContent": "center",
#                     "alignItems": "center",
#                     "height": "100vh",
#                 },
#             ),
#         ]
#     )

#     @app.callback(
#         [
#             Output("loss-plot", "figure"),
#             Output("lr", "children"),
#         ],
#         [Input("Learning rate", "value")],
#     )
#     def update_plot(lr: float):
#         fig = plot_training(lr)
#         return fig, f"Learning rate: {lr:.2f}"

#     return app
