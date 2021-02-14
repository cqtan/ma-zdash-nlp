import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./app/model/model_mlc_v7"
BATCH_SIZE = 32
SEQ_LEN = 200
THRESHOLD = 0.5
UNIQUE_LABELS = ["delivery", "feedback_return", "product", "monetary"]

# Performs multi-label classification to classify text as having
# one of the following types: delivery, returning, product, monetary
def run_multi_label_classification(df, col_text_input):
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
    tokenizer = AutoTokenizer.from_pretrained("CouchCat/ma_mlc_v7_distil")
    model = AutoModelForSequenceClassification.from_pretrained(
        "CouchCat/ma_mlc_v7_distil"
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
        with torch.no_grad():
            input_ids_batch = batch[0].to(device)
            output = model(input_ids_batch)

        pred_labels = torch.sigmoid(output[0]).to("cpu").numpy()
        predictions_list.extend(pred_labels.tolist())

    # Update dataframe
    col_return_list, col_deliver_list, col_product_list, col_monetary_list = (
        [],
        [],
        [],
        [],
    )
    for preds in predictions_list:
        col_return_list.append(True if preds[0] >= THRESHOLD else False)
        col_deliver_list.append(True if preds[1] >= THRESHOLD else False)
        col_product_list.append(True if preds[2] >= THRESHOLD else False)
        col_monetary_list.append(True if preds[3] >= THRESHOLD else False)

    df_out = df.copy()
    df_out["p_delivery"] = col_deliver_list
    df_out["p_return"] = col_return_list
    df_out["p_product"] = col_product_list
    df_out["p_monetary"] = col_monetary_list

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