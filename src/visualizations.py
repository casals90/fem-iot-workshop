from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def create_pie_chart_with_grouped_threshold(
        input_df: pd.DataFrame, column_name: str, ax, title: str,
        threshold: float = 0.01, grouped_label: str = 'Others',
        x_label: str = '', y_label: str = '', font_size: int = 14,
        start_angle: int = 90) -> None:
    """
    Create Pie chart from dataframe and grouped values by threshold.

    Args:
        input_df (pd.DataFrame): dataframe to plot.
        column_name (str): column to plot.
        ax (str): Matplotlib axes.
        title (str): plot's title.
        threshold (optional, float): threshold to apply. Default it is 0.01.
        grouped_label (optional, str): label value to put when grouped
            values are greater than a threshold.
            Default value is 'Others'
        x_label (optional, str): label for x axis.
        y_label (optional, str): label for y axis.
        font_size (optional, str): font size of plot.
        start_angle (optional, int): start angle of Pie chart.

    """
    grouped_df = input_df \
        .groupby(column_name) \
        .size() \
        .to_frame() \
        .reset_index() \
        .rename(columns={0: 'sizes'})

    if not grouped_df.empty:
        grouped_df['perc'] = grouped_df.sizes / grouped_df.sizes.sum()

        cond_lt_threshold = grouped_df.perc < threshold
        grouped_df.loc[cond_lt_threshold, 'label_cleaned'] = grouped_label
        grouped_df['label_cleaned'] = grouped_df \
            .label_cleaned \
            .fillna(grouped_df[column_name])

        grouped_df \
            .groupby('label_cleaned') \
            .sizes \
            .sum() \
            .sort_values() \
            .plot(kind='pie', autopct='%1.1f%%', title=title, ax=ax,
                  legend=None, xlabel=x_label, ylabel=y_label,
                  fontsize=font_size, startangle=start_angle)
    else:
        ax.remove()


def plot_measuring_points(
        df: pd.DataFrame, coordinates_column: str = 'coordinates',
        color_column: str = None) -> None:
    """
    Given a measuring points dataframe, coordinates column and color column,
    this function plots measuring points in a map. In addition, if it specifies
    a color column name, it plots points with same colors.

    Args:
        df (pd.DataFrame): measuring points dataframe to plot.
        coordinates_column (optional, str): coordinates dataframe's column name.
        color_column (optional, str): dataframe's column name to plot points with the same color.

    """
    df['lat'] = df[coordinates_column].str.split(',').str[0].astype('float')
    df['lon'] = df[coordinates_column].str.split(',').str[1].astype('float')

    try:
        measuring_point_ids = df['id']
    except KeyError:
        measuring_point_ids = df['measuring_point_id']

    df['text'] = measuring_point_ids + ' : ' + df['description']

    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color=color_column, hover_name='text',
        zoom=11, height=600)

    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'center': {'lon': 2.07, 'lat': 41.4},
            'style': "stamen-terrain",
            'zoom': 11})
    fig.show()


def plot_correlation_heat_map(
        input_df: pd.DataFrame, title: str, color_map='BrBG',
        fig_size: Tuple[int, int] = (25, 25),
        correlation_method: str = 'pearson') -> None:
    """
    Given a dataframe 'input_df', this function plots a correlation heat map
    between all available columns.

    Args:
        input_df (pd.DataFrame): a dataframe for which to plot the heat map
        title (str): heat map title
        color_map (str, optional): the colormap to use
        fig_size (Tuple[int, int], optional): plot's figure size
        correlation_method (str, optional): correlation method to apply

    """
    plt.figure(figsize=fig_size)

    mask = np.triu(np.ones_like(input_df.corr(), dtype=np.bool))
    heatmap = sns.heatmap(
        input_df.corr(method=correlation_method), mask=mask, vmin=-1,
        vmax=1, annot=True, cmap=color_map)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)


def plot_acf_pacf(
        timeseries: Union[pd.Series, pd.DataFrame], lags: int = None,
        fig_size: Tuple[int, int] = (25, 8)) -> None:
    fig = plt.figure(figsize=fig_size)
    fig.suptitle("ACF and PACF", fontsize=14)

    ax1 = fig.add_subplot(1, 2, 1)
    plot_acf(timeseries, ax=ax1, lags=lags)

    ax2 = fig.add_subplot(1, 2, 2)
    plot_pacf(timeseries, ax=ax2, lags=lags, method='ywm')
