import streamlit as st
import pandas as pd
import pickle
from scipy import spatial
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.express as px
# import umap
# from joblib import load
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# pipe_nn_encoder = load('../models/pipe_nn_encoder_03.joblib') 
# umap_mnist = load('../models/umap_mnist_03.joblib')
# process_text = st.sidebar.text_area('process_text',height=500)
# nn_represent_newtext = pipe_nn_encoder.predict(process_text)
# umap_projection_newtext = umap_mnist.transform(nn_represent_newtext)

library_select =  pd.read_pickle('../data/library_distances03.pkl')

# library_select_grouped = library_select.drop(columns=['id','title','authorcentury','proj_0','proj_1']).groupby('author').mean()
# dists = spatial.distance.pdist(library_select_grouped.values, metric='euclidean') # metric='cosine'
# mergings = linkage(dists, method='complete')
# plt.figure(figsize = (12,8))
# dendrogram(mergings,
#            labels = list(library_select_grouped.index),
#            leaf_rotation = 90,
#            leaf_font_size = 6)
# plt.tight_layout()

st.write('## Sklearn')

st.markdown(
        """
            Unfortunately, in order to produce decent results, the TfidfVectorizer (plus the  MLPClassifier) took about 2gb of space to save! So below is just a 2d projection of the 100-dimensional weights of the hidden layer of the MLPClassifier.
        """
    )

author_choice = st.multiselect('Choose the authors',
                                   options=library_select['author'].unique(),
                                   default=library_select['author'].unique()
                                   )

color_choice = st.radio(
    'Choose how to color the books:',
    ('By Author', 'By Century'))


fig = px.scatter(data_frame = library_select.loc[library_select['author'].isin(author_choice)],
                 x='proj_0',
                 y='proj_1',
                 width=1200,
                 height=800,
                 color= 'author' if color_choice == 'By Author' else 'authorcentury',
                 color_discrete_sequence=px.colors.qualitative.Light24, #.Alphabet, #Dark24
                 hover_data=['title']
                 )
fig.update_xaxes(visible=False, showticklabels=False)
fig.update_yaxes(visible=False, showticklabels=False)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
))

st.plotly_chart(fig, theme="streamlit", use_container_width=True)