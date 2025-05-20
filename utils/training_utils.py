from utils.source_code_feature_extractor import SourceCodeFeatureExtractor
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import pandas as pd 
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def  train_and_return_model(dataframe,model):
        codeFeatureExtractor = SourceCodeFeatureExtractor()
        codeFeatureExtractor.transform(dataframe)
        extra_features = dataframe[list(codeFeatureExtractor.features.keys())]
        extra_features_clean = extra_features.astype(float)
        X_extra = csr_matrix(extra_features_clean.values)

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = tfidf.fit_transform(dataframe['description_and_code'])

        X = hstack([X,X_extra])

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(dataframe['tags'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)

        return model


def analyze_tag_confusions(y_true, y_pred, ordered_multi_labeled_classes):
    confusions = {}

    for idx, tag in enumerate(ordered_multi_labeled_classes):
        cm = multilabel_confusion_matrix(y_true, y_pred)[idx]
        tn, fp, fn, tp = cm.ravel()
        
        if fn > 0:
            confused_with = []
            for i in range(len(y_true)):
                if y_true[i, idx] == 1 and y_pred[i, idx] == 0:
                    predicted_tags_indexes = np.where(y_pred[i] == 1)[0]
                    predicted_tags = [ordered_multi_labeled_classes[j] for j in predicted_tags_indexes if j != idx]
                    confused_with.extend(predicted_tags)

            if confused_with:
                confused_series = pd.Series(confused_with).value_counts()
                confusions[tag] = confused_series

    return confusions


def multilabel_stratified_split(X, y, test_size=0.2, random_state=42):
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test



def multilabel_oversample(X_train, y_train, min_count=50):
    label_counts = y_train.sum(axis=0).A1 if hasattr(y_train, "A1") else y_train.sum(axis=0)
    
    rare_labels = np.where(label_counts < min_count)[0]
    
    
    idx_to_duplicate = []
    for label in rare_labels:
        idxs = np.where(y_train[:, label] == 1)[0]
        idx_to_duplicate.extend(idxs)
    
    
    X_extra = X_train[idx_to_duplicate]
    y_extra = y_train[idx_to_duplicate]
    
    
    X_train_oversampled = vstack([X_train, X_extra])
    y_train_oversampled = np.vstack([y_train, y_extra])
    X_train_os, y_train_os = shuffle(X_train_oversampled, y_train_oversampled, random_state=42)

    return X_train_os, y_train_os
