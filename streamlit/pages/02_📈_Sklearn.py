import streamlit as st
import pandas as pd
import pickle
from scipy import spatial
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.express as px
import umap
from joblib import load
from sklearn.neural_network import MLPClassifier, MLPRegressor
from pathlib import Path

path = Path(__file__).parent.parent

pipe_nn_encoder = load(str(path / 'models/pipe_nn_encoder_01.joblib')) 
umap_mnist = load(str(path / 'models/umap_mnist_01.joblib'))

# from pathlib import Path

# path = Path(__file__).parent.parent / 'data/'
# st.write(path)

# process_text = st.sidebar.text_area('process_text',height=500)
# nn_represent_newtext = pipe_nn_encoder.predict(process_text)
# umap_projection_newtext = umap_mnist.transform(nn_represent_newtext)

library_distances2d =  pd.read_pickle(path / 'data/library_distances2d.pkl')
dist2d_mean = library_distances2d.iloc[0]['dist2d_mean']


st.write('## Sklearn')

st.markdown(
        """
            - Scroll down to see the 2d projection of the 21 authors and their books
            - Click on 'Plot Settings' to hide authors or to color based on century
            - Click on 'Add New Text' to see where it falls compared to the authors
        """
    )

with st.expander('Plot Settings'):
    author_choice = st.multiselect('Choose the authors',
                                    options=library_distances2d['author'].unique(),
                                    default=library_distances2d['author'].unique()
                                    )

    color_choice = st.radio(
        'Choose how to color the books:',
        ('By Author', 'By Century'))

with st.expander('Add New Text'):
    st.write('Please note that all fields below need to be filled out.')
    title_text = st.text_input('Title of Book or Text', value='')
    author_text = st.text_input('Author Name', value='')
    century_text = st.text_input('Author Century', value='21st century CE')
    process_text = st.text_area('Input Text')

if (title_text != '') & (author_text != '') & (century_text != '') & (process_text != ''):
    ## nn predict and then umap transform to 2d
    represent_text = pipe_nn_encoder.predict(pd.Series(process_text))
    represent_text = umap_mnist.transform(represent_text)

    plot_df =  pd.DataFrame({'title': title_text,
                            'author': author_text,
                            'authorcentury_str': century_text,
                            'proj_0': represent_text[:,0],
                            'proj_1': represent_text[:,1]})
    
    # to draw circle around the new point
    circle_edges = plot_df[['proj_0', 'proj_1']].copy()
    circle_edges['x0'] = circle_edges['proj_0'] - dist2d_mean
    circle_edges['x1'] = circle_edges['proj_0'] + dist2d_mean
    circle_edges['y0'] = circle_edges['proj_1'] - dist2d_mean
    circle_edges['y1'] = circle_edges['proj_1'] + dist2d_mean

    plot_df = pd.concat([library_distances2d.loc[library_distances2d['author'].isin(author_choice)], plot_df])

else:
    plot_df = library_distances2d.loc[library_distances2d['author'].isin(author_choice)]

fig = px.scatter(data_frame = plot_df,
                 x='proj_0',
                 y='proj_1',
                 width=1200,
                 height=800,
                 color= 'author' if color_choice == 'By Author' else 'authorcentury_str',
                 color_discrete_sequence=px.colors.qualitative.Light24, #.Alphabet, #Dark24
                 hover_data=['title']
                 )
fig.update_xaxes(visible=False, showticklabels=False)
fig.update_yaxes(visible=False, showticklabels=False)

if (title_text != '') & (author_text != '') & (century_text != '') & (process_text != ''):
    for i in range(circle_edges.shape[0]):
        fig.add_shape(type='circle',
            xref='x', yref='y',
            x0=circle_edges['x0'][i],
            y0=circle_edges['y0'][i],
            x1=circle_edges['x1'][i],
            y1=circle_edges['y1'][i],
            line_color='LightSeaGreen',
        )

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
))

st.plotly_chart(fig, theme="streamlit", use_container_width=True)