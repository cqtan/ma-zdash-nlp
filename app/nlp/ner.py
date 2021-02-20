import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "./app/model/model_ner_v6"
BATCH_SIZE = 32
SEQ_LEN = 200

# Using ner_v7 labels
UNIQUE_LABELS = [
    "B-BRND",
    "B-MATR",
    "B-MISC",
    "B-PERS",
    "B-PROD",
    "B-TIME",
    "I-BRND",
    "I-PERS",
    "O",
    "PAD",
]

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
    tokenizer = AutoTokenizer.from_pretrained("CouchCat/ma_ner_v7_distil")
    model = AutoModelForTokenClassification.from_pretrained("CouchCat/ma_ner_v7_distil")
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

    # Remove entries with no meaning
    lemmatizer = WordNetLemmatizer()
    tokens_filtered_list, labels_filtered_list = [], []
    for tokens, labels in zip(tokens_cleaned_list, labels_cleaned_list):
        tokens_filtered, labels_filtered = [], []
        for token, label in zip(tokens, labels):
            if label != "O" and token != "[PAD]" and token != "[SEP]":
                if (
                    label[0] == "I"
                    and len(labels_filtered)
                    and labels_filtered[-1][0] == "B"
                ):
                    # Combine tokens
                    tokens_filtered[-1] = f"{tokens_filtered[-1]}_{token}"
                else:
                    tokens_filtered.append(lemmatizer.lemmatize(token.lower(), "n"))
                    labels_filtered.append(label)

        tokens_filtered_list.append(tokens_filtered)
        # Remove prefix, e.g. "B-", "I-"
        labels_clean = [label[2:] for label in labels_filtered]
        labels_filtered_list.append(labels_clean)

    # Create new dataframe for entity frequencies
    entity_freq = get_entity_frequency(tokens_filtered_list)
    labels2ents = get_labels2ents(tokens_filtered_list, labels_filtered_list)
    df_new = pd.DataFrame()
    df_new["entity"] = entity_freq.keys()
    df_new["freq"] = entity_freq.values()

    # Update dataframe
    df_out = df.copy()
    df_out["entities"] = tokens_filtered_list
    df_out["ent_labels"] = labels_filtered_list

    return df_out, df_new, labels2ents


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


# Lists all entities with their frequencies
def get_entity_frequency(entities_list):
    entity_freq = dict()
    for entities in entities_list:
        for entity in entities:
            if entity in entity_freq:
                entity_freq[entity] += 1
            else:
                entity_freq[entity] = 1

    return entity_freq


# List of all unique entities per label
def get_labels2ents(ents_list, labels_list):
    unique_labels = set(list(itertools.chain(*labels_list)))
    labels2ents = dict(zip(unique_labels, [[] for _ in range(len(unique_labels))]))

    for ents, labels in zip(ents_list, labels_list):
        for ent, label in zip(ents, labels):
            labels2ents[label].append(ent)

    return labels2ents
