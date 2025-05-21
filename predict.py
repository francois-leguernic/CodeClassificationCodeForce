import argparse
from utils.training_config import TrainingConfig
from utils.preprocessing_utils import build_dataframe_from_json, parse_data
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import time 
import numpy as np 

def measure_prediction_time(model, X, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        start = time.time()
        _ = model.predict(X)
        end = time.time()
        times.append(end - start)
    
    avg_time_total = np.mean(times)
    print(f"Average  time for {X.shape[0]} samples : {avg_time_total:.4f} s")
    
def main():
    parser = argparse.ArgumentParser(description="Test model predictions and inference time")
    parser.add_argument("--input_model_dir", type=str, help="Path of the dir where is stored the trained model")
    args = parser.parse_args()
    
    
    config = TrainingConfig(tags=['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities'])
    data = build_dataframe_from_json(parse_data(),config)
    
    
    model,vectorizer = joblib.load(os.path.join(args.input_model_dir,"model.joblib")),joblib.load(os.path.join(args.input_model_dir,"vectorizer.joblib"))
    X = vectorizer.transform(data['description_and_code'])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['tags'])
    y_pred = model.predict(X)
    
    measure_prediction_time(model,X)
    print(classification_report(y,y_pred,target_names=mlb.classes_))
    
    
if __name__=="__main__":
    main()