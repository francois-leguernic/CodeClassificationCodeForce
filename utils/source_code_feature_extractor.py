import pandas as pd 
import re   

class SourceCodeFeatureExtractor:
    def __init__(self):
        self.features = {
            #"has_math_terms": (self.has_math_terms_in_code,"source_code"),
            "num_loops": (self.count_loops,"source_code"),
            "has_comparison_operator": (self.has_comparison_operator,"source_code"),
            #"has_probability_terms_in_desc":(self.has_probability_terms,"prob_desc_description"),
            #"has_geometry_keywords":(self.contains_geometry_keywords,"source_code")
            "uses_modulo" : (self.uses_modulo,"source_code"),
            "has_bitwise" : (self.uses_bitwise,"source_code"),
            "uses_gcd" : (self.uses_gcd,"source_code"),
            "uses_lcm" : (self.uses_lcm,"source_code"),
            "uses_prime" : (self.uses_prime_words,"source_code")
        }

        self.features_to_keep = set()
    
    def add_feature_to_keep(self,featureName):
        self.features_to_keep.add(featureName)
    
    def get_training_features(self):
        return list(set(self.features.keys()).union(self.features_to_keep))

    def has_recursion(self,text):
        print("-------")
        print(text)
        fname = text.split("def")[1].split("(")[0].strip()
        is_recursive = text.count(fname + "(") > 1

        return is_recursive

    def count_loops(self,text):
        return len(re.findall(r"\bfor\b|\bwhile\b", text))

    def has_comparison_operator(self,text):
        return any(op in text for op in [">", ">=", "<=", "<"])
    
    def has_math_terms_in_code(self,code):
    
        code = code.lower()
        math_patterns = [
            r'\bmath\.', 
            r'\bsqrt\b',
            r'\bpow\b',
            r'\bmod\b|\bmodulo\b|\b%\b',
            r'\bgcd\b',
            r'\blcm\b',
            r'\bprime\b',
            r'\bfactorial\b',
            r'\bdivisor\b',
            r'\bcomb\b|\bperm\b',
            r'\blog\b',
            r'\bceil\b|\bfloor\b',
            r'\bsin\b|\bcos\b|\btan\b',
            r'\bpi\b|\be\b',
            r'\babs\b',
            r'\bround\b',
            r'\bint\(',
            r'[\+\-\*/\^]=?',
        ]

        for pattern in math_patterns:
            if re.search(pattern, code):
                return 1

        return 0
    
    def has_probability_terms(self,text):
        terms = [
            r"\bprobability\b", r"\bexpected\b", r"\bexpectation\b", r"\bvariance\b",
            r"\bdistribution\b", r"\brandom\b", r"\brandomly\b", r"\brandomized\b",
            r"\bsimulation\b", r"\bsimulate\b", r"\bmonte carlo\b", r"\buncertain\b",
            r"\bexpected value\b", r"\bmean\b", r"\bstandard deviation\b",
            r"\bstochastic\b", r"\bprobabilistic\b", r"\bwith probability\b",
            r"\bexpected number\b",r"\bbernoulli\b",r"\bgaussian\b",r"\bchance\b"
            ,r"\bdice\b",r"\bcoin\b",r"\buniform\b"
        ]
        pattern = re.compile("|".join(terms), re.IGNORECASE)
        return int(bool(pattern.search(text)))


    def transform(self, df, col="source_code"):
        for name, (func,col_name) in self.features.items():
            df[name] = df[col_name].apply(func)
        return df

    def contains_geometry_keywords(self,text):

        GEOMETRY_KEYWORDS = ['point', 'points', 'coordinates', 'polygon', 'plane', 'line', 'segment', 'segments',
        'x', 'y', 'r', 'radius', 'distance', 'area', 'triangle', 'circle','sqrt', 'acos', 'atan2', 'math.pi'
        'cos','sin','asin']

        text = str(text).lower()
        return int(any(kw in text for kw in GEOMETRY_KEYWORDS))


    def uses_modulo(self, text):
        return bool(re.search(r"%|\bmod\b", text))

    def uses_gcd(self, code):
        return bool(re.search(r"\b(?:__)?gcd\s*\(", code))

    def uses_lcm(self, code):
        return bool(re.search(r"\blcm\s*\(", code))

    def uses_pow(self, code):
        return bool(re.search(r"\*\*|\bpow\s*\(", code))

    def uses_bitwise(self, code):
        return bool(re.search(r"\b(?:<<|>>)\b|[&|^~]", code))

    def uses_divisors(self, code): 
        return bool(re.search(r"\bfor\s+\w+\s+in\s+range\s*\(.*?//.*?\)", code))

    def uses_prime_words(self, code): 
        return bool(re.search(r"\bis_prime\b|\bprime\b", code, flags=re.I))
    