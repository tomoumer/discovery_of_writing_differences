import streamlit as st
from pathlib import Path

path = Path(__file__).parent.parent

st.write('## Summary')

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
            Overall I was able to get to my goal - identify some differences between authors and be able to introduce new text to compare.

            A few key points:
            - Even a simple bag of words approach is able to decently distinguish authors (given enough books)
            - Taking the trained weights of the last hidden layer and projecting onto a 2d plane is often able to cluster together books belonging to same author
            - Using a pretrained Hugging Face transformer with only five books per author, the results are even better (splitting each book in 10 parts)
        """
    )

with col2:
    st.image(str(path / 'img/journal.png'), use_column_width='auto')

st.markdown(
    """
        Possible extensions:
        - Even better cleanup of the text (e.g., removing the introduction)
        - Adding even more authors
        - For Sklearn:
            - splitting the books into multiple parts (like with Hugging Face)
        - For Hugging Face:
            - selecting books that are long enough to permit more than 10 parts of 512 tokens
            - using another model (besides bert base uncased)
    """
)