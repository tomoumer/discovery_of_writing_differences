import streamlit as st
import pandas as pd
import plotly.express as px

import random

pielovers = ['great', 'more great', 'greater', 'more great than great', 'less great, but still more', 'the most great ever', 'there is nothing greater than a pie chart', 'pie charts are life', 'pie chart, the universe, and everything', 'mathematics in pie chart dimensions', 'pie chart for life', 'the greatest',
             'great: the greatening']

randomlist = [random.random() for i in range(len(pielovers))]

df = pd.DataFrame({'category': pielovers,
                   'values': randomlist})

st.markdown(
    """
    This page is here just to say **thank you** to everyone at Nashville Software School.

    In my view, teachers and professors often don't receive enough credit for the tremendous work that they are doing.

    Thank you, **Michael Holloway**, our instructor extraodrinaire and our two wonderful TAs, **Rohit Venkat** and **Neda Taherkhani**.

    This has been an incredible, interesting and fun 9 months. Thank you for all your patience and for sharing the knowledge.

    And to showcase my gratitude, here's showing just how much I learned! Here is **the best** graph kind that exists - a pie chart. Just for you, Michael! 
"""
)


fig = px.pie(df, title='How great are pie charts, really?',
             values='values', names='category',
             width=800, height=800,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')


st.plotly_chart(fig, theme="streamlit", use_container_width=True)