import pygwalker as pyg
from dash import Dash, html
import dash_dangerously_set_inner_html
import pandas as pd


def render_df(df: pd.DataFrame) -> Dash:
    """
    Uses the pygwalker library to render a DataFrame as an HTML table in Dash.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to render.

    Returns
    -------
    Dash
        The Dash app with the DataFrame rendered as an HTML table.
    """

    walker = pyg.walk(df, debug=False)
    html_code = walker.to_html()

    app = Dash()

    app.layout = html.Div(
        [dash_dangerously_set_inner_html.DangerouslySetInnerHTML(html_code)]
    )

    return app
