import streamlit as st
import plotly.graph_objects as go
import app.charts.constants as con


def render_single_entity_by_sen(df, labels2ents, label_type):
    st.subheader(f"Entity by Sentiment ({label_type})")
    entities = set(list(labels2ents[label_type]))
    ent2sents = dict(zip(entities, [[0, 0, 0] for _ in range(len(entities))]))

    df_ents = df[df["entities"].astype(str) != "[]"]  # Filtering out empty lists
    if len(df_ents):
        ents = df_ents["entities"].astype(str).tolist()
        for i, ent in enumerate(ents):
            ent = ent[1:-1].replace("'", "").replace(" ", "")
            ent_list = ent.split(",")

            for e in ent_list:
                if e in ent2sents:
                    if df_ents["p_sentiment"].iloc[i] == "negative":
                        ent2sents[e][0] += 1
                    elif df_ents["p_sentiment"].iloc[i] == "neutral":
                        ent2sents[e][1] += 1
                    else:
                        ent2sents[e][2] += 1

        x = list(ent2sents.keys())
        y = list(ent2sents.values())

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
