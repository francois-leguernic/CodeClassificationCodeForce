import pandas as pd
import tqdm 
import os 
import json 
import numpy as np
import random
import sys
from utils.training_config import TrainingConfig

def parse_data():
    dataset_dir = "code_classification_dataset"
    files = os.listdir(dataset_dir)
    jsonList = []

    for f in tqdm.tqdm(files):
        with open(os.path.join(os.getcwd(),dataset_dir,f),'r',encoding="utf8") as jsonFile:
            data = json.load(jsonFile)
            jsonList.append(data)
    return jsonList

def get_raw_features(jsonElement):
    return [prop for prop in jsonElement]

def get_raw_columns(jsonElement:dict):
    return [value for key,value in jsonElement.items()]

def build_dataframe_from_json(json:list,config:TrainingConfig):
    properties = get_raw_features(json[0])
    data = [get_raw_columns(jsonElement) for jsonElement in json]

    dataframe = pd.DataFrame(columns=properties,data=data)
    dataframe["description_and_code"] = dataframe["prob_desc_description"] + " [SEP] " + dataframe["source_code"]
    
    dataframe["tags"] = dataframe["tags"].apply(lambda tags: filter_selected_tags(tags, config.tags))

    dataframe = dataframe[dataframe["tags"].map(len) > 0]
    return dataframe 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_rows_containing_label(label,dataframe):
    return dataframe[dataframe["tags"].apply(lambda tags : label in tags)]

def filter_selected_tags(row, selected_tags):
    return [tag for tag in row if tag in selected_tags]
