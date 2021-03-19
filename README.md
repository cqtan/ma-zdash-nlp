[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/couchcat/ma-zdash-nlp/main)

# zDash NLP

## Master Thesis Prototype

![Screenshot 2021-03-03 at 22 40 49](https://user-images.githubusercontent.com/33485290/111790183-3b02bc00-88c2-11eb-928c-05cd1380ab50.png)

## Demo

- View prototype on Streamlit: https://share.streamlit.io/couchcat/ma-zdash-nlp/main
- Trained models available here: https://huggingface.co/CouchCat

## Description

Gathering large amounts of annotated textual data by hand or by statistical means is a tedious and time consuming process. The following demo application utilizes a light version of the popular BERT architecture known as DistilBERT to perform a small number of Natural Language Processing (NLP) tasks to infer insights on customer feedback texts. This architecture allows good confidence in its predictions with the use of a relatively small number of training data. Insights gained are illustrated as a collection of charts (dashboard) resulting from the following tasks:

- Sentiment Analysis (negative, neutral, positive)
- Multi-Label Classification (Theme of the text)
  - Delivery (delivery status, time of arrival, etc.)
  - Return (return confirmation, return label requests, etc.)
  - Product (quality, complaint, etc.)
  - Monetary (pending transactions, refund, etc.)
- Named Entity Recognition (Extraction of certain words)
  - Product (shoe, hat, bag, etc.)
  - Brand (Nike, Adidas, Armani, etc.)

Results are highly tied to the kind of data the application is trained on, which makes it fairly context dependent to its domain and tasks. While its accuracy could still be improved the main goal was to quickly prototype an application that does not require much training resources in order to achieve usable results in an efficient manner and to evaluate its potential.

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
