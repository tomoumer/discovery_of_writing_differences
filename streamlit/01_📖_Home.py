import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

authors_df = pd.read_pickle('../data/select_authors.pkl')
library_en = pd.read_pickle('../data/library_en.pkl')
authors_pg_links = ''

authors_pg_links = ''

for author in authors_df['author']:
    authors_pg_links += f'- [{author}]'
    authors_pg_links += f'(https://www.gutenberg.org/ebooks/search/?query={author}&submit_search=Go%21)\n'.replace(', ', '%20').replace(' ', '%20')

st.set_page_config(
    page_title='TU_capstone_DS6',
    page_icon='📖',
)


st.write('# Discovery of Writing Differences')
st.write('A capstone Data Science project by **Tomo Umer, MS**.')
st.divider() 

col1, col2 = st.columns([1, 3])

with col1:
    st.image('https://tomoumerdotcom.files.wordpress.com/2022/04/cropped-pho_logo_notext.png', use_column_width='auto', caption='VERBA VOLANT, SCRIPTA MANENT')

with col2:
    st.markdown(
        """
            Welcome to my Capstone Project for the [6th cohort of Data Science](https://nss-data-science-cohort-6.github.io/#) with the [Nashville Software School](https://nashvillesoftwareschool.com)!

            I've always loved stories in any shape or form (books, movies, games, whatever) and am a self-published writer - [my webpage if you are curious](https://tomoumer.com).
            
            Click on various tabs below to learn more about the project, the reasoning and the choices I made along the way*. Otherwise, dive straight into the demonstrations by navigating the sidebar.

            *no dragons or other mythological creatures were harmed in producing this
        """
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Background', 'Idea', 'Cleanup', 'Authors', 'Methodology'])

#with st.expander('The Background'):
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
        """
            To me, the story of human civilization is that of languages. Having words and creating abstract concepts is what allowed us to develop a mathematic framework upon which the rest of the sciences have been built.
            
            With data science and machine learning, we've come full circle and are able to use math to analize the languages that gave rise to it in the first place. And of course, the answer to life, the universe and everything is 42.
        """
        ) 
    with col2:
        st.image('../img/civ1_alphabet.png', use_column_width='auto', caption="Sid Meier's Civilization")

    col1, col2 = st.columns([1,5])
    with col1:
        st.image('https://www.gutenberg.org/gutenberg/pg-logo-129x80.png', use_column_width='auto')
    with col2:
        st.write("The [Project Gutenberg](https://www.gutenberg.org) represents a fantastic collection of over 70,000+ books through millenia, so that was an obvious place to start. The caveat there is that the site contains only works in public domain (deceased authors with copyrights expired)."
        )

    st.markdown(
        """
            To be completely honest, if finances were not an issue, I'd rather buy and use works from currently living writers in order to promote them on top of doing a fun project. Regardless, I think Alan Moore (one of the greatest writers of our time) sums up the essence of writing and reading phenomenally in his BBC Maestro Course:

            > Writing has been, and always will be, our foremost means of modifying human consciousness. You are modifying the consciousness of the reader and therefore you are modifying the reality of the reader. Writing will modify the reality and the consciousness of the entire species and, inevitably, will mean modifying the consciousness of the writer themselves.         
        """
    )

#with st.expander('The Idea'):
with tab2:
    st.markdown(
        """
            Some years ago as I picked up word crafting again, I was told by a friend that my writing reminds him of another author. That is something that stuck with me as fascinating, since I haven't actually read any works from that particular author (nor can I remember who it was, sorry).
        """
    )
    st.image('../img/mybooks.png', use_column_width='auto', caption='books from my collection')
    
    st.markdown(
        """
           When I'm deciding on what to read next, there are two factors that come into play: theme and writing style.
           
           The theme is easily understood and can also be extracted using Natural Language Processing (though ultimately did not end up being part of my final project). It can represent abstract topics, like philosophy, stoicism or mathematics. It can also be much more concrete, like seeing a book covers that captivates us, or knowing that a book has a phoenix in it (10/10 for me, right away).
           
           I would argue that with a good writer, they could be describing the most awfully boring of topics and yet still retain most people's attention. And that is because of the choice of words, the structure of their writing and the overall style. And this writing style (or writing differences when comparing authors) is what I was curious to try and capture with my capstone project.

           One of the starting points for this project was the paper on the [stylometric approach](https://aclanthology.org/2020.wnut-1.30.pdf).
        """
    )

# with st.expander('The Cleanup (and The Resulting Authors)'):
with tab3:
    st.markdown(
        """
            This is where challenges started. There were some issues that I was anticipating and others that I did not. At all. I used the [standardized Gutenberg corpus](https://github.com/pgcorpus/gutenberg) to download the corpus between Apr 8-14, 2023. There is a metadata file that I used as reference, called `library`. This is a random sample (after selecting only english books, see below):
        """
    )

    st.dataframe(library_en.drop(columns=['authorcentury', 'authorcentury_str']).sample(5), use_container_width=True)

    st.markdown(
        """
            There was a lot of back and forth, continuously updating and refining what I did. Feel free to dig through the files on [my git repository](https://github.com/tomoumer/discovery_of_writing_differences), but I'll outline some of the main parts below. Starting with initial filters, to get the workable books:

            - select books (exclude other file formats, like images)
            - remove 'index of Gutenberg' content (it's just links)
            - take english language only - which can mean translations
            - ignore books that haven't been downloaded (not sure why, less than 2000)
            - remove unknown, anonymous or varied authors
         """
    ) 

    st.write('Number of available English books is', len(library_en), 'written by', library_en['author'].nunique(), 'authors. The total number of English book downloads by known authors over the past 30 days was', library_en['downloads'].sum())

    st.divider() 

    st.markdown("""
        ### Which Authors to Choose?
    
        Due to hardware limitations I knew I wouldn't be able to simply use all of the available books. Below are the top 10 most prolific authors and breakdown of released books by centuries that authors lived in.
    """)

    col1, col2 = st.columns([3,1])

    with col1:
        # seaborn option:
        # fig = plt.figure(figsize = (10,6))
        # sns.barplot(data= (library_en
        #                 .groupby('author')[['title']]
        #                 .count()
        #                 .sort_values(by='title', ascending=False)
        #                 .rename(columns={'title': 'num_books'})
        #                 .head(10)
        #                 .reset_index()),
        #             x='num_books',
        #             y='author')
        # st.pyplot(fig)
        fig = px.bar((library_en
                        .groupby('author')[['title']]
                        .count()
                        .sort_values(by='title', ascending=False)
                        .rename(columns={'title': 'num_books'})
                        .head(10)
                        .reset_index()),
                        x='num_books', y='author', orientation='h')
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with col2:
        st.dataframe(library_en.groupby(['authorcentury','authorcentury_str'])[['id']]
            .count()
            .reset_index()
            .drop(columns='authorcentury')
            .rename(columns={'authorcentury_str': 'century','id': 'num_books'}),
            hide_index=True
        )

    st.markdown("""
        I did some additional cleanup and exploration, also considering the downloads over past 30 days.

        Ultimately, I decided to go with a collection of 21 authors over centuries, and whose writing I was familiar with (see 'authors' tab). Depending on the algorithm used, I also needed to limit the amount of books for some of the most prolific authors.

        Finally, I read in the books and successively removed the author's names from the beginning of each book.

        ```
        def import_book(filepath):
            try:
                with open(filepath, encoding = 'utf-8') as fi:
                    book = fi.read()
            except:
                with open(filepath, encoding = 'unicode_escape') as fi:
                    book = fi.read()

            if(not re.search('\*\*\*\ START OF .+? \*\*\*', book)):
                book_start = 0
            else:
                book_start = re.search('\*\*\* START OF .+? \*\*\*', book).end()

            if(not re.search('\*\*\*\ END OF .+? \*\*\*', book)):
                book_end = -1
            else:
                book_end = re.search('\*\*\* END OF .+? \*\*\*', book).start()

            book = book[book_start : book_end]

            return book
        ```

        The more I worked on this, the more avenues for improvement I discovered. For example, there are books where *** are encoded in a strange way and the above function doesn't catch them. I did not have the time to explore this further. 
    """)

with tab4:
    st.write('Selected authors with number of available books:')
    #st.bar_chart(library_en.loc[library_en['author'].isin(authors_df['author'])].groupby('author')[['id']].count().rename(columns={'id': 'num_books'}))
    fig = px.bar((library_en
                  .loc[library_en['author'].isin(authors_df['author'])]
                  .groupby(['author', 'authorcentury', 'authorcentury_str'])[['id']]
                  .count()
                  .reset_index()
                  .sort_values(by='authorcentury')
                  .rename(columns={'id': 'num_books'})),
                x='author', y='num_books', color='authorcentury_str')
    #fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    # st.write('Here is the full list of authors and their respective centuries, considered for this project.')
    # st.table(authors_df.drop(columns=['authorcentury','author_num']).set_index(np.arange(1, len(authors_df) + 1)))

    st.write('For anybody interested, here are direct links to the Gutenberg Project search for the authors.')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(authors_pg_links)

    with col2:
        st.image('../img/books_history.png', use_column_width='auto', caption='fun with Midjourney - love the double pointed pens!')

#with st.expander('The Methodology'):
with tab5:
    st.markdown(
        """
            I was able to use two distinct libraries for this project, [scikit-learn](https://scikit-learn.org/stable/) and then [Hugging Face](https://huggingface.co).

            ### Scikit-Learn

            Limiting to at most 50 books per writer I used the following pipeline:

            ```
            pipe_nn = Pipeline(
                steps = [
                    ('vect', TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))),
                    ('scaler', MaxAbsScaler()),
                    ('nn', MLPClassifier(verbose = True, hidden_layer_sizes = (100, 100)))
                ]
            )
            ```
            
            1. TfidfVectorizer splits the text into words or couples of words (unigrams and bigrams)
                - CountVectorizer turns the words into vectors (bag of words)
                - TfidfTransformer normalizes the word counts accross documents
            2. Scale feature by its maximum absolute value (helps the ml algorithm converge faster)
            3. At first I used a logistic regression to test and then passed on to MLPClassifier with two hidden layers of 100 nodes each

            After the model was trained:
            - Copy the MLPClassifier into a MLPRegressor
            - Take only the second to last layer (last hidden layer of 100 nodes)
            - Calculate distances between books or authors
            - In order to display it, used UMAP to reduce dimensionality from 100 to 2
            
            One such representation is available under 'Sklearn' tab.

            ### Hugging Face

            The base idea is still similar to above. The difference is that with Hugging Face, you can start with a pretrained model which is more complex than just simple words.

            Bert Base (uncased) model splits words into sub-words at times. It is limited to 512 tokens, which typically correspond to about 400 words (give or take). For this reason, to try and make things equal, I took 5 books per author and split each book into 10 parts (ignoring the beginning and end of the book).

            I did not have the time to look into how to take the last hidden layer for a representation like I've done previously. Still, I incorporated the saved model into this app, the 'Hugging Face' tab. It uses the weights obtained during training to best estimate who the given writing is similar to.
        """
    )

    # insert example matrices above

    # st.latex(r"""
    #     tfidf(t, d) = tf(t, d) * (log \frac{1 + n}{1 + df(t)} + 1)
    # """)


st.divider() 