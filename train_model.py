from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from utils.preprocessing_utils import build_dataframe_from_json, parse_data
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer,MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score
from scipy.sparse import hstack, csr_matrix
from utils.training_config import TrainingConfig
from utils.source_code_feature_extractor import SourceCodeFeatureExtractor
from utils.training_utils import multilabel_stratified_split 
from lightgbm import LGBMClassifier
import datetime
import os


class ModelTrainer:
    def __init__(self, model = None, param_grid=None, use_grid_search=False, cv=3, scoring="f1_macro"):
        self.model = model or LGBMClassifier(class_weight='balanced',verbose=-1,random_state=42)
        self.param_grid = param_grid
        self.use_grid_search = use_grid_search
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.best_f1_score_macro = None
        self.tfidvectorizerText = None
        self.tfidvectorizerCode = None
        self.labelvectorizer = None
        self.minmaxscaler = None 

    def train(self, X_train, y_train):
        print("Training model...")
        clf = OneVsRestClassifier(self.model)

        if self.use_grid_search and self.param_grid:
            print("Using GridSearchCV...")
            clf = GridSearchCV(clf, self.param_grid, cv=self.cv, scoring=self.scoring, verbose=2, n_jobs=-1)

        clf.fit(X_train, y_train)

        if self.use_grid_search:
            print("Best Params:", clf.best_params_)
            self.best_model = clf.best_estimator_
        else:
            self.best_model = clf

        print("Training complete.")
        return self.best_model

    def evaluate(self, X_test, y_test, label_names=None):
        if not self.best_model:
            raise Exception("Model not trained yet. Call train() first.")
        y_pred = self.best_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_names)
        print(report)
        f1_score_macro = f1_score(y_test,y_pred,average="macro")
        print(f"Best F1 score macro {f1_score_macro}")
        return report

    def save_model(self, model_type):
        if not self.best_model:
            raise Exception("Model not trained yet. Call train() first.")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("models", f"{model_type}_{timestamp}")
        model_path = os.path.join(model_dir,"model.joblib")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.best_model, model_path)
        vectorizer_text_path= os.path.join(model_dir,"vectorizer_text.joblib")
        joblib.dump(self.tfidvectorizerText, vectorizer_text_path)
        vectorizer_code_path= os.path.join(model_dir,"vectorizer_code.joblib")
        joblib.dump(self.tfidvectorizerText, vectorizer_code_path)
        label_vectorizer_path = os.path.join(model_dir,"labelVectorizer.joblib")
        joblib.dump(self.labelvectorizer,label_vectorizer_path)
        min_max_scaler_path = os.path.join(model_dir,"minmaxscaler.joblib")
        joblib.dump(self.minmaxscaler,min_max_scaler_path)

        print(f"Model saved to {model_path}")

    
    def save_datasets(self,df_train df_test):



def main():
    parser = argparse.ArgumentParser(description="Train a multilabel model  TF-IDF.")
    parser.add_argument("--gridsearch", action="store_true", help="Enable GridSearchCV")
    parser.add_argument("--extra_features",action="store_true",help="Aditional features to concatenate to TF-IDF vectors")
    parser.add_argument("--model_type",help="Model type that will be serialized")
    args = parser.parse_args()
    
    config = TrainingConfig(tags=['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities'])
    
    df = build_dataframe_from_json(parse_data(), config)
    
    tfidf_text = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), stop_words='english', lowercase=True, 
                            max_df=0.8, min_df=5, sublinear_tf=True, strip_accents='unicode')

    X_text = tfidf_text.fit_transform(df['prob_desc_description'])

    tfidf_code = TfidfVectorizer(max_features=300, ngram_range=(1, 3),stop_words='english',lowercase=True,max_df=0.8,
        min_df=5, sublinear_tf=True, strip_accents='unicode')
    
    X_code = tfidf_code.fit_transform(df['source_code'])
    
    
    scaler = MinMaxScaler()
    df["scaled_difficulty"] = scaler.fit_transform(df[['difficulty']])
     
    if args.extra_features:
        codeFeatureExtractor = SourceCodeFeatureExtractor()
        codeFeatureExtractor.transform(df)
        codeFeatureExtractor.add_feature_to_keep("scaled_difficulty")
        extra_features = df[codeFeatureExtractor.get_training_features()]
        print(f"adding extra features {codeFeatureExtractor.get_training_features()}")
        extra_features_clean = extra_features.astype(float)
        X_extra = csr_matrix(extra_features_clean.values)
        X = hstack([X_text,X_code,X_extra])
    else:
        X = hstack([X_text,X_code])


    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tags"])

   
    X_train, X_test, y_train, y_test = multilabel_stratified_split(X, y)

    param_grid = {
        "estimator__n_estimators": [100, 300],
        "estimator__max_depth": [6, 10],
        "estimator__learning_rate": [0.05, 0.1],
    } if args.gridsearch else None

    trainer = ModelTrainer(use_grid_search=args.gridsearch, param_grid=param_grid)
    trainer.tfidvectorizerText = tfidf_text
    trainer.tfidvectorizerCode = tfidf_code
    trainer.labelvectorizer = mlb
    trainer.minmaxscaler = scaler
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test, label_names=mlb.classes_)
    trainer.save_model(args.model_type)
    
    

if __name__ == "__main__":
    main()

