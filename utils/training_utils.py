from source_code_feature_extractor import SourceCodeFeatureExtractor
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

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
                    predicted_tags = np.where(y_pred[i] == 1)[0]
                    predicted_tags = [ordered_multi_labeled_classes[j] for j in predicted_tags if j != idx]
                    confused_with.extend(predicted_tags)

            if confused_with:
                confused_series = pd.Series(confused_with).value_counts()
                confusions[tag] = confused_series

    return confusions
