import streamlit as st
import numpy as np
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
from scipy import special

#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


authors_df = pd.read_pickle('data/select_authors.pkl')

select_authors = authors_df['author'].to_list()

checkpoint = 'bert-base-uncased'

# note truncation side and padding side are to determine which side to cutoff - beginning (left) or end (rigt)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation_side='right', padding_side='right')

def tokenize_function(df):
    return tokenizer(df['text'], truncation=True, padding='max_length',  max_length=512)

acc = evaluate.load('accuracy') #average = None
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')
mcc = evaluate.load('matthews_correlation')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc_m = acc.compute(predictions=predictions, references=labels)
    precision_m = precision.compute(predictions=predictions, average = 'macro', references=labels)
    recall_m = recall.compute(predictions=predictions, average = 'macro', references=labels)
    f1_m = f1.compute(predictions=predictions, average = 'macro', references=labels)
    mcc_m = mcc.compute(predictions=predictions, references=labels)
    metrics = {
        'accuracy': acc_m['accuracy'],
        'precision': precision_m['precision'],
        'recall': recall_m['recall'],
        'f1': f1_m['f1'],
        'mcc': mcc_m['matthews_correlation']
    }
    return metrics

model = AutoModelForSequenceClassification.from_pretrained('../models/bert_base_uncased/fivebooks_tenparts/')

test_args = TrainingArguments(
    output_dir= '../models/bert_base_uncased/fivebooks_tenparts/',
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=4,
    use_mps_device=True
)

trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=compute_metrics
)


st.write('## Hugging Face')

st.write("""
    - Copy & paste or write the text in the box below
    - The text gets processed in chunks of 512 tokens (about 2500 characters or 400 words)
    - There may be some text cut off at the end of each 2500 character set
    - It will take increasingly more time depending on the length of the text prompt.
    """
)
         
process_text_hf = st.text_area('Text to Process')

if len(process_text_hf) % 2500 != 0:
    st.write('The length of the text to process is', len(process_text_hf), 'which corresponds to:', len(process_text_hf) // 2500 + 1, 'chunks')
else:
    st.write('The length of the text to process is', len(process_text_hf), 'which corresponds to:', len(process_text_hf) // 2500, 'chunks')

st.divider()

# this next function to split the text into chunks of 2500
def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def compute_winners(model_predictions):
    unique_num, counts = np.unique(model_predictions.argmax(axis=1), return_counts=True)

    unique_authors = [authors_df.loc[authors_df['author_num'] == unique, 'author'].iloc[0] for unique in unique_num]

    return pd.DataFrame({'Most Likely Author': unique_authors, 'Number of Chunks':counts})

if process_text_hf != '':
    serialized_text = pd.Series(chunkstring(process_text_hf, 2500))

    newtext_ds = Dataset.from_dict({'text': serialized_text})
    tokenized_newtext_ds= newtext_ds.map(tokenize_function)
    tokenized_newtext_ds = tokenized_newtext_ds.remove_columns(['text'])

    # get predictions
    new_results = trainer.predict(tokenized_newtext_ds) 

    # calculate who "wins" over chunks

    st.write('These are the overall results:')
    author_winners = compute_winners(new_results.predictions)
    st.dataframe(author_winners)


    # calculate actual probabilities
    new_probabilities = special.softmax(new_results.predictions, axis=1)

    # get probabilities and authors in a dataframe and transpose it - to be able to use .head()
    new_probabilities_df = pd.DataFrame(new_probabilities, columns=select_authors).T

    for i, column in enumerate(new_probabilities_df):
        with st.expander(f'Chunk number {column +1}'):
            st.dataframe(new_probabilities_df[column].sort_values(ascending=False).head(7).rename('Likelihood').map('{:.2%}'.format))
            st.write('Processed Text Chunk:')
            st.write(f'{serialized_text.iloc[i]}')

            #col3, col4 = st.columns([1,2])
            #with col1:
            #    st.dataframe(new_probabilities_df[column].sort_values(ascending=False).head(7).rename('Likelihood').map('{:.2%}'.format))
            #with col2:
            #    st.write('Processed Text Chunk:')
            #    st.write(f'{serialized_text.iloc[i]}')