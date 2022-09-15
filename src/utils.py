import pandas as pd


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
