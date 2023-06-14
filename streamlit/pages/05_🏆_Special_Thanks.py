import streamlit as st
import pandas as pd
import plotly.express as px

import random

pielovers = ['because', 'because because', 'because x5', 'they are greater than life', 'pie charts are less great, but still more', 'what are pie charts even', 'there is nothing greater than a pie chart', 'pie charts are life', 'pie chart, the universe, and everything', 'mathematics is just pie charts', 'pie chart for life', 'pie charts are pioneers', 'can you eat a pie chart though?']

randomlist = [random.random() for i in range(len(pielovers))]

df = pd.DataFrame({'category': pielovers,
                   'values': randomlist})

st.write('## Thank you Nashville Software School!!')

st.image("https://nashvillesoftwareschool.com/images/NSS-logo-horizontal-small.jpg", use_column_width='auto')

st.markdown(
    """
    This page is an acknowledgment and thank you to everyone at Nashville Software School. I can only imagine how much work got put into making this opportunity possible. From program developers to the marketing team, from everyone I had the pleasure of meeting virtually, to those that I didn't.

    In particular though, what would a school be without teachers? So thank you:
    - **Michael Holloway**, our instructor extraodrinaire
    - our two wonderful TAs, **Rohit Venkat** and **Neda Taherkhani**
    
    This has been an incredible, interesting and fun 9 months under your guidance.
    
    And finally, thank you to my awesome 15 classmates. A classroom is only as alive as the people in it and I've had a bunch of great conversations, exchanging of ideas and discussions with many of you.

    To showcase my gratitude, and the depth of my obtained knowledge, here is **the best** graph kind that exists. A pie chart - just for you, Michael! 
"""
)


fig = px.pie(df, title='Why are pie charts the best?',
             values='values', names='category',
             width=800, height=800,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')


st.plotly_chart(fig, theme="streamlit", use_container_width=True)