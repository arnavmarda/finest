import pygwalker as pyg
from dash import Dash, html
import dash_dangerously_set_inner_html
import pandas as pd
from rich import print as rprint
from typing import List
from src.indicators.news import SearchResult


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


def pprint_search_results(results: List[SearchResult]) -> None:
    """
    Function to pretty print the search results.

    Parameters
    ----------
    results : List[SearchResult]
        List of search results to pretty print.

    Returns
    -------
    None
    """
    for i, r in enumerate(results):
        rprint(f"[bold]Result {i+1}[/bold]")
        rprint("-" * 50)
        rprint(f"[red]Provider: {r.provider}[/red]")
        rprint(f"[blue]Title: {r.title}[/blue]")
        rprint(f"[green]URL: {r.url}[/green]")
        rprint(f"[yellow]Text: {r.text}[/yellow]")
        rprint("\n")
        rprint("\n")
