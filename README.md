- To train and save a model, you can use the script train_model.py, which saves the best OneVsRestClassified, composed of LightGBM : 

e.g. : ``` python train_model.py  --model_type lgbm  --gridsearch```

- To make predictions on the whole dataset and measure the average prediction time, you can use the script predict.py :

 e.g : ``` python predict.py --input_model_dir "C:\Users\francoisle-guernic\Desktop\CodeClassification\models\lgbm_20250521_105331" ```