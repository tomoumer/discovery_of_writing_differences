import streamlit as st
import pandas as pd
import numpy as np
import pickle

authors_df = pd.read_pickle('../data/select_authors.pkl')

st.set_page_config(
    page_title='TU_capstone_DS6',
    page_icon='📖',
)


st.write('# Discovery of Writing Differences')
st.write('A capstone Data Science project by **Tomo Umer, MS**.')
st.divider() 

col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://tomoumerdotcom.files.wordpress.com/2022/04/cropped-pho_logo_notext.png", use_column_width='auto', caption='VERBA VOLANT, SCRIPTA MANENT')

with col2:
    st.markdown(
        """
            Welcome to my Capstone Project for the [6th cohort of Data Science](https://nss-data-science-cohort-6.github.io/#) with the [Nashville Software School](https://nashvillesoftwareschool.com)!

            I've always loved stories in any shape or form (books, movies, games, whatever) and am a self-published writer - [my webpage if you want to read](https://tomoumer.com).
            
            You can expand various tabs below to learn more about the background to the project, the reasoning and the choices I made along the way (no dragons or other mythological creatures were harmed in producing this). Otherwise, dive straight into the demonstrations by clicking on the options in the sidebar.
        """
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Authors', 'Background', 'Idea', 'Cleanup', 'Methodology'])


with tab1:
    #st.dataframe(authors_df.drop(columns='authorcentury').reset_index(drop=True))
    st.write('Here is the full list of authors and their respective centuries, considered for this project.')
    st.table(authors_df.drop(columns=['authorcentury','author_num']).set_index(np.arange(1, len(authors_df) + 1)))

#with st.expander('The Background'):
with tab2:
    st.markdown(
        """
            To me, the story of human civilization is that of languages. Having words and creating abstract concepts is what allowed us to develop a mathematic framework upon which the rest of the sciences have been built. With data science and machine learning, we've come full circle and are able to use math to analize the languages that gave rise to it in the first place. And of course, the answer to life, the universe and everything is 42.

            And the [Project Gutenberg](https://www.gutenberg.org) represents a fantastic collection of over 70,000+ books through millenia, so that was an obvious place to start. To be completely honest though, if I had money, I'd buy and use works from currently living authors in order to promote them as well as do a fun project.

            Regardless, I think Alan Moore (one of the greatest authors of our time), sums up the essence of writing and reading phenomenally:

            > Writing has been, and always will be, our foremost means of modifying human consciousness. You are modifying the consciousness of the reader and therefore you are modifying the reality of the reader. Writing will modify the reality and the consciousness of the entire species and, inevitably, will mean modifying the consciousness of the writer themselves.         
        """
    )

#with st.expander('The Idea'):
with tab3:
    st.markdown(
        """
            Some years ago as I picked up word crafting again, I was told by a friend that my writing reminds him of another author. That is something that stuck with me as fascinating, since I haven't actually read any works from that particular author (nor can remember who it was, sorry).
        
            Also for myself, when deciding on what to read next, there are two factors that come into play: theme and writing style. The theme is of course very easily understood (and can be deduced using nlp algorithms). Theme can be very abstract, like philisophy (or even more specific, like Stoicism), or very much concrete, like seeing a book covers that captivates you, or knowing that a book has a phoenix in it (10/10 for me, right away).
            
            I would argue however that with a good writer, they could be writing the most awfully boring of topics and yet still retain most people's attention. And that is because of the choice of words, the structure of their writing and the overall style. And this, more elusive idea is what I was curious to try and capture with my capstone project.
        """
    )

# with st.expander('The Cleanup (and The Resulting Authors)'):
with tab4:
    st.markdown(
        """
            This is where challenges started. There were some issues that I was anticipating and others that I did not. At all.

            First of all, using the [standardized Gutenberg corpus](https://github.com/pgcorpus/gutenberg) I was able to download almost all content. There were some other files (like images and movies), as well as some files missing. I did not end up spending a whole lot of time focussing on why certain books did not download.

            From there it was a lot of back and forth, continuously updating and refining what I did. Feel free to dig through the files on [my git repository](https://github.com/tomoumer/discovery_of_writing_differences), but the TL;DR version is here:

            - select only English books
            - finalize which authors to use (21, ultimately a combination of living century + my personal choice)
            - eliminate additional junk books (like Gutenberg index of, which is just an index)
            - for extremely prolific authors, limit books to 50 (for huggingface model actually only 5 for each author)
            - read books in using regex to get rid of the Gutenberg info at beginning and end
            - delete authors names from the books (to not allow ML to train on authors names within introduction or such)

            The more I worked on this, the more avenues for improvement I discovered.
        """
    ) 

#with st.expander('The Methodology'):
with tab5:
    st.markdown(
        """
            When starting this project, I thought I would begin with [topic modelling](https://arxiv.org/pdf/2103.00498.pdf), but instead, I ended up more closely following the idea behind a [stylometric approach](https://aclanthology.org/2020.wnut-1.30.pdf).

            To start with, I used [scikit-learn](https://scikit-learn.org/stable/). The very first iteration of the program involved a TfidfVectorizer and Logistic regression as proof of concept. That one is not included in this app. A much better (and more complex) iteration followed. Still started with the TfidfVectorizer (n-gram range 1 or 2) and then passing the results through a MaxAbsScaler and finally a MLPClassifier with two hidden layers of 100 nodes each.

            Next, instead of using those predictions directly, I copied the trained model into a MLPClassifier, stopping at the second to last layer (the last hidden layer of 100 nodes). That allowed me to essentially obtain a 100-dimensional representation for each book. That can then be used to calculate distances between various works. Further, in order to represent the work on a neat plot, I used UMAP to reduce from 100 dimensions to 2. One such representation is available under 'Sklearn' tab.

            Finally, I attempted a Hugging Face model, by using a Bert Base (uncased) model as a starting point. For this portion I had to resort to using google colab for the training due to offering GPU-acceleration and fed it the same authors. The difference from before is however, these models can't take more than a certain number of characters - 512 in this case. So I decided to take 5 books from each author (the least amount of books all authors have), and take 512 characters at 10 different points in the book - ignoring the very beginning and end.

            Once that was trained, I incorporated the saved model into this app, the 'Hugging Face' tab. It uses the weights obtained during training to best estimate who the given writing is similar to.
        """
    )

st.divider() 