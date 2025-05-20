import argparse
from utils.training_config import TrainingConfig
from utils.preprocessing_utils import build_dataframe_from_json, parse_data

def main():
    parser = argparse.ArgumentParser(description="Test model predictions and inference time")
    parser.add_argument("--input_model", type=str, help="Path for trained model")
    args = parser.parse_args()
    
    
    config = TrainingConfig(tags=['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities'])
    data = build_dataframe_from_json(parse_data(),config)
    

if __name__=="__main__":
    main()