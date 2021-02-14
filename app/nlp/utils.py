import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def get_bert_base_model_and_tokenizer():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    MODEL_PATH = ""
    BATCH_SIZE = 32
    SEQ_LEN = 200
    UNIQUE_LABELS = ["negative", "neutral", "positive"]

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", do_lower_case=True
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(UNIQUE_LABELS)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)