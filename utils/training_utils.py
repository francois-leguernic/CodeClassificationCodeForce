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
import matplotlib.pyplot as plt

"""
CLI for training model 
"""
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

"""
Returns all the tags that are present in a false-negative prediction for all the tags
"""
def analyze_tag_confusions(y_true, y_pred, ordered_multi_labeled_classes):
    confusions = {}
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    
    for idx, tag in enumerate(ordered_multi_labeled_classes):
        cm = confusion_matrix[idx]
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


def bar_plot_confusion(confusion_dico):
    labels = list(confusion_dico.keys())
    fig,ax = plt.subplots(len(labels),1,figsize=(15,5*len(labels)))

    for i in range(len(labels)):
        confusions = confusion_dico[labels[i]]
        ax[i].set_title(f"Confusion for label {labels[i]}")
        ax[i].bar(list(confusions.index),list(confusions.values))
        ax[i].set_xticklabels(confusions.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def multilabel_stratified_split(X, y,all_indices,test_size=0.2, random_state=42):
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(X, y):
        train_idx = np.array(all_indices)[train_idx]
        test_idx = np.array(all_indices)[test_idx]
        return train_idx, test_idx
    return 


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

def reduce_math(df, target_count=150):
    import pandas as pd 
    only_math = df[df['tags'].apply(lambda tags: tags == ['math'])]
    mixed_math = df[df['tags'].apply(lambda tags: 'math' in tags and tags != ['math'])]
    others = df[df['tags'].apply(lambda tags: 'math' not in tags)]

    
    reduced_only_math = only_math.sample(n=target_count, random_state=42)
    df_reduced = pd.concat([reduced_only_math, mixed_math, others], ignore_index=True)
    return df_reduced
