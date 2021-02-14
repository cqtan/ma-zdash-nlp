import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./app/model/model_sa_v7"
BATCH_SIZE = 32
SEQ_LEN = 200
UNIQUE_LABELS = ["negative", "neutral", "positive"]

# Performs sentiment analysis to predict either "negative", "neutral", "positive"
# sentiments for each text input
def run_sentiment_analysis(df, col_text_input):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ########## For loading the models locally #########
    # tokenizer = DistilBertTokenizer.from_pretrained(
    #     "distilbert-base-uncased", do_lower_case=True
    # )
    # model = DistilBertForSequenceClassification.from_pretrained(
    #     "distilbert-base-uncased", num_labels=len(UNIQUE_LABELS)
    # )

    # # Load pre-trained model weights
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    ######### Loading uploaded models from Huggingface #########
    tokenizer = AutoTokenizer.from_pretrained("CouchCat/ma_sa_v7_distil")
    model = AutoModelForSequenceClassification.from_pretrained(
        "CouchCat/ma_sa_v7_distil"
    )
    model.to(device)

    # Get texts from dataframe
    texts = df[col_text_input].astype(str).tolist()

    # Prepare DataLoader
    input_ids_list = get_input_ids_list(tokenizer, texts)
    predict_data = TensorDataset(torch.tensor(input_ids_list))
    predict_sampler = SequentialSampler(predict_data)
    predict_dataloader = DataLoader(
        predict_data, sampler=predict_sampler, batch_size=BATCH_SIZE
    )

    # Get model predictions
    predictions_list = []
    model.eval()
    for batch in predict_dataloader:
        input_ids_batch = batch[0].to(device)
        with torch.no_grad():
            output = model(input_ids_batch)
        predictions_list.extend(
            np.argmax(F.softmax(output[0], dim=1).cpu().detach().numpy(), axis=1)
        )

    # Update dataframe
    predicted_sentiments = [UNIQUE_LABELS[pred] for pred in predictions_list]
    df_out = df.copy()
    df_out["p_sentiment"] = predicted_sentiments

    return df_out


# Vectorize texts for BERT
def get_input_ids_list(tokenizer, texts):
    input_ids_list = []
    for text in texts:
        input_ids = tokenizer.encode(text)
        input_ids.extend([0] * SEQ_LEN)
        input_ids = input_ids[:SEQ_LEN]
        input_ids_list.append(input_ids)

    return input_ids_list