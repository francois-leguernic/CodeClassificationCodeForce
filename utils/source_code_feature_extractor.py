import pandas as pd 
import re   

class SourceCodeFeatureExtractor:
    def __init__(self):
        self.features = {
            "has_recursion": self.has_recursion,
            "num_loops": self.count_loops,
            "has_comparison_operator": self.has_comparison_operator,
            
        }

    def has_recursion(self,text):
        fname = text.split("def")[1].split("(")[0].strip()
        is_recursive = text.count(fname + "(") > 1

        return is_recursive

    def count_loops(self,text):
        return len(re.findall(r"\bfor\b|\bwhile\b", text))

    def has_comparison_operator(self,text):
        return any(op in text for op in [">", ">=", "<=", "<"])

    def transform(self, df, col="description_and_code"):
        for name, func in self.features.items():
            df[name] = df[col].apply(func)
        return df