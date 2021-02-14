import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
)

MODEL_SA = "./app/models/model_sa_v7"
MODEL_MLC = "./app/models/model_mlc_v7"
MODEL_NER = "./app/models/model_ner_v6"

MODEL_SA_OUT = "./out/sa/"
MODEL_MLC_OUT = "./out/mlc/"
MODEL_NER_OUT = "./out/ner/"


########### SA ###########
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", do_lower_case=True
)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
model.load_state_dict(torch.load(MODEL_SA, map_location=torch.device("cpu")))
model.save_pretrained(MODEL_SA_OUT)
tokenizer.save_pretrained(MODEL_SA_OUT)
# tf_model = TFDistilBertForSequenceClassification.from_pretrained(
#     MODEL_SA_OUT, from_pt=True
# )
# tf_model.save_pretrained(MODEL_SA_OUT)


########### MLC ###########
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", do_lower_case=True
)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)
model.load_state_dict(torch.load(MODEL_MLC, map_location=torch.device("cpu")))
model.save_pretrained(MODEL_MLC_OUT)
tokenizer.save_pretrained(MODEL_MLC_OUT)


########### NER ###########
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-cased", do_lower_case=False
)
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-cased", num_labels=4
)
model.load_state_dict(torch.load(MODEL_NER, map_location=torch.device("cpu")))
model.save_pretrained(MODEL_NER_OUT)
tokenizer.save_pretrained(MODEL_NER_OUT)
