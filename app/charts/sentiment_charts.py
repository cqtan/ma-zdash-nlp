import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import app.charts.constants as con


def render_sen_by_cat(df):
    st.subheader("Sentiment by Category")
    cat_by_sen = dict(zip(con.kpis, [[], [], [], []]))

    for col in con.kpis:
        for sen in con.sentiments:
            total = len(df[df[col] == 1])
            df_tmp = df[df[col] == 1]
            num_sen = len(df_tmp[df_tmp["p_sentiment"] == sen])
            percentage = 0
            if num_sen != 0:
                percentage = round(num_sen / total * 100, 1)
            cat_by_sen[col].append(percentage)

    fig = go.Figure()
    rows = list(cat_by_sen.values())

    for i in range(0, len(con.sentiment_labels)):
        for xd, yd in zip(rows, con.kpi_labels):
            fig.add_trace(
                go.Bar(
                    x=[xd[i]],
                    y=[yd],
                    orientation="h",
                    marker=dict(
                        color=con.colors_sentiment[i],
                        line=dict(color="rgb(248, 248, 249)", width=1),
                    ),
                )
            )
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.1, 1],
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode="stack",
        height=con.chart_height,
        paper_bgcolor=con.paper_bgcolor,
        plot_bgcolor=con.paper_bgcolor,
        margin=con.chart_margins,
        showlegend=False,
    )

    annotations = []

    for yd, xd in zip(con.kpi_labels, rows):
        # labeling the y-axis
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.1,
                y=yd,
                xanchor="right",
                text=str(yd),
                font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
            )
        )
        # labeling the first percentage of each bar (x_axis)
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=xd[0] / 2,
                y=yd,
                text=str(xd[0]) + "%",
                font=dict(family="Arial", size=14, color="rgb(248, 248, 255)"),
                showarrow=False,
            )
        )
        # labeling the first Likert scale (on the top)
        if yd == con.kpi_labels[-1]:
            annotations.append(
                dict(
                    xref="x",
                    yref="paper",
                    x=xd[0] / 2,
                    y=-0.1,
                    text=con.sentiment_labels[0],
                    font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                    showarrow=False,
                )
            )
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=space + (xd[i] / 2),
                    y=yd,
                    text=str(xd[i]) + "%",
                    font=dict(family="Arial", size=14, color="rgb(248, 248, 255)"),
                    showarrow=False,
                )
            )
            # labeling the Likert scale
            if yd == con.kpi_labels[-1]:
                annotations.append(
                    dict(
                        xref="x",
                        yref="paper",
                        x=space + (xd[i] / 2),
                        y=-0.1,
                        text=con.sentiment_labels[i],
                        font=dict(family="Arial", size=14, color="rgb(67, 67, 67)"),
                        showarrow=False,
                    )
                )
            space += xd[i]

    fig.update_layout(annotations=annotations)

    st.plotly_chart(fig)


def render_sen_by_time(df):
    st.subheader("Sentiment (Across Months)")
    sen_by_month = dict()
    for i, month in enumerate(con.months):
        df_tmp = df.loc[(pd.to_datetime(df["dt"])).dt.month == i]
        if len(df_tmp):
            sen_values = []
            for sen in con.sentiments:
                sen_values.append(len(df_tmp[df_tmp["p_sentiment"] == sen]))
            sen_by_month[month] = sen_values

    months_available = list(sen_by_month.keys())
    values_by_month = list(sen_by_month.values())
    fig = go.Figure()
    for i in range(0, len(con.sentiments)):
        fig.add_trace(
            go.Bar(
                x=months_available,
                y=[val[i] for val in values_by_month],
                name=con.sentiment_labels[i],
                marker_color=con.colors_sentiment[i],
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


def sen_by_count_donut(df):
    st.subheader("Sentiment Distribution")
    sen_values = []
    for sen in con.sentiments:
        sen_values.append(len(df[df["p_sentiment"] == sen]))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=con.sentiment_labels,
                values=sen_values,
                hole=0.5,
                marker_colors=con.colors_sentiment,
            )
        ]
    )
    fig.update_layout(
        xaxis=dict(
            domain=[0, 1],
        ),
        height=con.chart_height,
        width=con.chart_height + 140,
        paper_bgcolor=con.paper_bgcolor,
        margin=con.chart_margins,
        showlegend=True,
    )
    st.plotly_chart(fig)
