import json
import os

import numpy
import pandas as pd


def convert(o):
    if isinstance(o, numpy.int64) or isinstance(o, numpy.int32):
        return int(o)
    raise TypeError

def dump_json(file_path, data, indent=4, sort_keys=True):
    with open(os.path.join(file_path), 'w') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, default=convert)
        f.close()

def dump_json_to_csv(output_path, data):
    pd.DataFrame(data).to_csv(output_path, encoding='utf-8', index=False)

def get_json(file_path, create_if_not_exist=False):
    try:
        with open(os.path.join(file_path), encoding="utf8") as f:
            response = json.load(f)
            f.close()
            return response
    except FileNotFoundError as e:
        if create_if_not_exist is True:
            with open(os.path.join(file_path), "w+") as f:
                json.dump({}, f)
            with open(os.path.join(file_path),  encoding="utf8") as f:
                response = json.load(f)
                f.close()
                return response
        else:
            raise e

def get_xlsx(file_path, column_by=None):
    try:
        xlsx_as_pandas = pd.read_excel(file_path)
    except:
        xlsx_as_pandas = pd.read_csv(file_path)
    if column_by is not None:
        xlsx_as_pandas = xlsx_as_pandas.set_index(column_by)
    xlsx_as_json_str = xlsx_as_pandas.to_json()
    xlsx_as_json = json.loads(xlsx_as_json_str)
    return xlsx_as_json

def get_txt(file_path):
    f = open(file_path, "r")
    return f.read()

def dump_csv(csv_path, csv):
    pd.DataFrame(csv).to_csv(csv_path, encoding='utf-8')

