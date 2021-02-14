import requests
import pandas as pd
import os.path
from io import StringIO


def read_csv_from_google_drive(link):
    # URL = "https://docs.google.com/uc?export=download&id="
    # url = requests.get(URL + file_id).text
    # csv = StringIO(url)
    URL = "https://docs.google.com/uc?export=download"

    file_id = link.split("/")[-2]
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    csv = StringIO(response.text)

    return csv


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def get_limit(limit):
    if limit == "Sample (5)":
        return 5
    elif limit == "complete":
        return None
    else:
        return int(limit)


def read_csv_to_config(isLocal, file, limit, col_text, col_date, col_others=""):
    df = None
    if isLocal:
        file_name = f"./app/data/{file}"
    else:
        file_name = read_csv_from_google_drive(file)

    cols = [col_text, col_date]
    if len(col_others) > 1:
        cols.extend(col_others.split(","))

    nrows = get_limit(limit)
    df_raw = pd.read_csv(file_name, nrows=nrows)

    if nrows == None:
        df = df_raw
    else:
        df = df_raw.sample(nrows, random_state=42)

    return dict(df=df, col_text=col_text, col_date=col_date, total_rows=len(df_raw))


def check_local_availability(file_name):
    return os.path.isfile(f"./app/data/{file_name}")
