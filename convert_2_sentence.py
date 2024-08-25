
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


data = pd.read_csv("#path/to/data")


data['Filename'] = data['Filename'].astype(str)
data['Token'] = data['Token'].astype(str)

data = data.rename(columns = {"Filename": "Sentence #","Token": "Word" ,"BIO": "Tag"})



print("Number of tags: {}".format(len(data.Tag.unique())))
frequencies = data.Tag.value_counts()


label2id = {k: v for v, k in enumerate(data.Tag.unique())}
id2label = {v: k for v, k in enumerate(data.Tag.unique())}



def create_sentence_df(df):
    sentences_data = []
    current_sentence = []
    current_tags = []
    current_file = None

    for _, row in df.iterrows():
        token = row['Word']
        bio_tag = row['Tag']
        file_name = row['Sentence #']
        
        if file_name != current_file:
            if current_sentence:  
                sentences_data.append((current_file, ' '.join(current_sentence), ','.join(current_tags)))
                current_sentence = []
                current_tags = []
            current_file = file_name  

        current_sentence.append(token)
        current_tags.append(bio_tag)
        
        if token.endswith('.') or token.endswith('!') or token.endswith('?'):
            sentences_data.append((current_file, ' '.join(current_sentence), ','.join(current_tags)))
            current_sentence = []
            current_tags = []

    if current_sentence:
        sentences_data.append((current_file, ' '.join(current_sentence), ','.join(current_tags)))
        
    return pd.DataFrame(sentences_data, columns=['Filename', 'sentence', 'word_labels'])


data = create_sentence_df(data)

#data = data[["sentence", "word_labels"]]
data['word_labels'] = data['word_labels'].astype(str)
data.to_csv("#path/to/output_data", index = False)
data.head(31)

len(data)


