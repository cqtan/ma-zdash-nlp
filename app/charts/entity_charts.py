import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import app.charts.constants as con


def render_entity_by_sen(df, entities):
    st.subheader("Entity by Sentiment")
    df_ents = df[df["entities"].astype(str) != "[]"]  # Filtering out empty lists
    if len(df_ents):
        ents = df_ents["entities"].astype(str).tolist()
        ent2val = dict(zip(entities, [[0, 0, 0] for _ in range(len(entities))]))
        for i, ent in enumerate(ents):
            ent = ent[1:-1].replace("'", "").replace(" ", "")
            ent_list = ent.split(",")
            for e in ent_list:
                if df_ents["p_sentiment"].iloc[i] == "negative":
                    ent2val[e][0] += 1
                elif df_ents["p_sentiment"].iloc[i] == "neutral":
                    ent2val[e][1] += 1
                else:
                    ent2val[e][2] += 1

        x = list(ent2val.keys())
        y = list(ent2val.values())

        fig = go.Figure()
        for i, sen in enumerate(con.sentiment_labels):
            fig.add_trace(
                go.Bar(
                    name=sen,
                    x=x,
                    y=[values[i] for values in y],
                    marker_color=con.colors_sentiment[i],
                )
            )

        fig.update_layout(
            barmode="stack",
            xaxis_tickangle=-45,
            xaxis=dict(domain=[0, 1], categoryorder="total descending"),
            height=con.chart_height,
            width=1200,
            paper_bgcolor=con.paper_bgcolor,
            margin=con.chart_margins,
            showlegend=True,
        )

        st.plotly_chart(fig)
    else:
        st.warning("No entities recognized")
