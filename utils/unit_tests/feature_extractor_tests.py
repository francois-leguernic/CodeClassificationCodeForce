import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import source_code_feature_extractor

class TestFeatures(unittest.TestCase):

    def test_is_code_recursive(self):
        sourceCode = """
        def factorial(n)
                    if n == 0:
                    return 1
        return n * factorial(n - 1) """

        extractor = source_code_feature_extractor.SourceCodeFeatureExtractor()
        result = extractor.has_recursion(sourceCode)
        self.assertEqual(result,1)

    def test_is_not_code_recursive(self):
        sourceCode = """
        def function(n):
                if n == 0:
                  return 1 # function is a dumb function
        return 2 """

        extractor = source_code_feature_extractor.SourceCodeFeatureExtractor()
        result = extractor.has_recursion(sourceCode)
        self.assertEqual(result,0)

if __name__ == '__main__':
    unittest.main()



        