import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "./app/model/model_ner_v6"
BATCH_SIZE = 32
SEQ_LEN = 200
UNIQUE_LABELS = ["BRND", "O", "PAD", "PRD"]

# Performs named entity recognition using DistilBERT
# specifically for words relating to brands and products
def run_named_entity_recognition(df, col_text_input):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ########## For loading the models locally #########
    # tokenizer = DistilBertTokenizer.from_pretrained(
    #     "distilbert-base-cased", do_lower_case=False
    # )

    # # This one uses the Token classifier version
    # model = DistilBertForTokenClassification.from_pretrained(
    #     "distilbert-base-cased", num_labels=len(UNIQUE_LABELS)
    # )

    # # Load pre-trained model weights
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    ######### Loading uploaded models from Huggingface #########
    tokenizer = AutoTokenizer.from_pretrained("CouchCat/ma_ner_v6_distil")
    model = AutoModelForTokenClassification.from_pretrained("CouchCat/ma_ner_v6_distil")
    model.to(device)

    # Get texts from dataframe
    texts = df[col_text_input].astype(str).tolist()

    # Prepare DataLoader
    input_ids_list, tokens_list = get_input_ids_and_tokens_list(tokenizer, texts)
    predict_data = TensorDataset(torch.tensor(input_ids_list))
    predict_sampler = SequentialSampler(predict_data)
    predict_dataloader = DataLoader(
        predict_data, sampler=predict_sampler, batch_size=BATCH_SIZE
    )

    # Get model predictions
    predicted_ids_list = []
    model.eval()
    for batch in predict_dataloader:
        with torch.no_grad():
            input_ids_batch = batch[0].to(device)
            output = model(input_ids_batch)
        predictions = np.argmax(output[0].to("cpu").numpy(), axis=2)
        predicted_ids_list.extend(predictions.tolist())

    # Combine sub-words
    tokens_cleaned_list, labels_cleaned_list = [], []
    for tokens, label_ids in zip(tokens_list, predicted_ids_list):
        tokens_cleaned, labels_cleaned = [], []
        for token, id in zip(tokens, label_ids):
            if token.startswith("##"):
                tokens_cleaned[-1] = tokens_cleaned[-1] + token[2:]
            else:
                labels_cleaned.append(UNIQUE_LABELS[id])
                tokens_cleaned.append(token)

        tokens_cleaned_list.append(tokens_cleaned)
        labels_cleaned_list.append(labels_cleaned)

    # Remove entries with no meaning and get unique entries
    lemmatizer = WordNetLemmatizer()
    tokens_unique_list, labels_unique_list = [], []
    for tokens, labels in zip(tokens_cleaned_list, labels_cleaned_list):
        tokens_filtered, labels_filtered = [], []
        for token, label in zip(tokens, labels):
            if label != "O" and token != "[PAD]" and token != "[SEP]":
                tokens_filtered.append(lemmatizer.lemmatize(token.lower(), "n"))
                labels_filtered.append(label)

        tokens_unique_list.append(list(set(tokens_filtered)))
        labels_unique_list.append(list(set(labels_filtered)))

    # Create new dataframe for entity frequencies
    entity_freq = get_entity_frequency(tokens_unique_list)
    df_new = pd.DataFrame()
    df_new["entity"] = entity_freq.keys()
    df_new["freq"] = entity_freq.values()

    # Update dataframe
    df_out = df.copy()
    df_out["entities"] = tokens_unique_list

    return df_out, df_new


# Vectorize texts and retrieve word tokens for BERT NER task
def get_input_ids_and_tokens_list(tokenizer, texts):
    input_ids_list, tokens_list = [], []
    for text in texts:
        input_ids = tokenizer.encode(text)
        input_ids.extend([0] * SEQ_LEN)
        input_ids = input_ids[:SEQ_LEN]
        input_ids_list.append(input_ids)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tokens.extend([0] * SEQ_LEN)
        tokens = tokens[:SEQ_LEN]
        tokens_list.append(tokens)

    return input_ids_list, tokens_list


def get_entity_frequency(entities_list):
    entity_freq = dict()
    for entities in entities_list:
        for entity in entities:
            if entity in entity_freq:
                entity_freq[entity] += 1
            else:
                entity_freq[entity] = 1

    return entity_freq