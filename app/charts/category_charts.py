import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import charts.constants as con

# Categories count chart
def render_cat_by_count(df):
    st.subheader("Text Categories (Total Counts)")
    cat_values = []
    for kpi in con.kpis:
        kpi_len = len(df[df[kpi] == 1])
        cat_values.append(kpi_len)

    fig = go.Figure(
        go.Bar(
            x=cat_values,
            y=con.kpi_labels,
            orientation="h",
            marker_color=con.colors,
        )
    )
    fig.update_layout(
        xaxis=dict(
            domain=[0, 1],
        ),
        height=con.chart_height,
        paper_bgcolor=con.paper_bgcolor,
        margin=con.chart_margins,
        showlegend=False,
    )
    st.plotly_chart(fig)


# Categories by time (month)
def render_cat_by_time(df):
    st.subheader("Text Categories (Across Months)")
    categories_by_month = dict()
    for i, month in enumerate(con.months):
        df_tmp = df.loc[(pd.to_datetime(df["dt"])).dt.month == i]
        if len(df_tmp):
            category_values = []
            for kpi in con.kpis:
                category_values.append(len(df_tmp[df_tmp[kpi] == 1]))
            categories_by_month[month] = category_values

    months_available = list(categories_by_month.keys())
    values_by_month = list(categories_by_month.values())
    fig = go.Figure()
    for i, kpi in enumerate(con.kpis):
        fig.add_trace(
            go.Bar(
                x=months_available,
                y=[val[i] for val in values_by_month],
                name=con.kpi_labels[i],
                marker_color=con.colors[i],
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        xaxis=dict(
            domain=[0, 1],
        ),
        height=con.chart_height,
        paper_bgcolor=con.paper_bgcolor,
        margin=con.chart_margins,
        showlegend=True,
    )
    st.plotly_chart(fig)
