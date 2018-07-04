import os
import argparse
import difflib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine, jaccard, hamming, euclidean, dice

def analyzer(s: str):
    assert type(s) == str

    tokens = list()
    curr_token = ''
    for c in s:
        if not c.isalnum():
            if curr_token:
                tokens.append(curr_token)
                curr_token = ''
            if not c in [' ', '(', ')', ',']:
                tokens.append(c)
            continue
        curr_token += c
    return tokens

def compute_score(TSNB, TSB, LS, TI, LI, TL, LL):
    from math import sin, cos
    from operator import add, sub, mul, neg
    
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    div = lambda a, b: protectedDiv(a, b)
    
    # GP-learned model (Learning-to-Rank)
    # This is dummy, not the final one
    return neg(add(mul(TSB, sub(sin(neg(add(TSNB, TSB))), LS)), LI))

def predict_unit(ingredient: str, source: list):
    assert type(ingredient) == str
    assert type(source) == list and all(type(line) == str for line in source)

    # Use sklearn's CountVectorizer for BOW
    vectorizers = [
        CountVectorizer(ngram_range=(1,1), lowercase=False, binary=False, analyzer=analyzer),
        CountVectorizer(ngram_range=(1,1), lowercase=False, binary=True, analyzer=analyzer)
    ]

    vectors = [ vectorizer.fit_transform([ingredient] + source).toarray() for vectorizer in vectorizers ]

    ingredient_vectors = [ v[0] for v in vectors ]
    source_vectors = [ v[1:] for v in vectors ]

    features = list()
    max_num_toks, max_len = 0, 0
    for i in range(len(source)):
        line = source[i]

        # Cosine Similarity bewteen BOW vectors
        bowsims = [ cosine(ingredient_vectors[v], source_vectors[v][i]) for v in range(len(vectorizers)) ]
        for i, bowsim in enumerate(bowsims):
            if np.isnan(bowsim):
                bowsims[i] = 1.0
        
        # Lexical Similarity
        seq_matcher = difflib.SequenceMatcher(lambda x: x == " ", ingredient, line)
        lexsim = 1 - seq_matcher.ratio()
        
        if len(analyzer(line)) > max_num_toks:
            max_num_toks = len(analyzer(line))
        if len(line) > max_len:
            max_len = len(line)
        
        features.append(bowsims + [lexsim, len(analyzer(ingredient)), len(ingredient), len(analyzer(line)), len(line)])
    
    for row in features:
        row[-4] /= float(max_num_toks)
        row[-3] /= float(max_len)
        row[-2] /= float(max_num_toks)
        row[-1] /= float(max_len)

    values = [compute_score(*row) for row in features]
    min_value = min(values)
    candidates = list()
    for j, value in enumerate(values):
        if value == min_value:
            candidates.append(j)

    # Return the middle one if many candidates exist
    return candidates[len(candidates)//2] + 1

def predict(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        ingredient = lines[0].strip()
        source = list(map(lambda l: l.rstrip(), lines[2:]))
        return predict_unit(ingredient, source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()
    
    if args.v:
        correct, incorrect = 0, 0

    directory = os.fsencode(os.path.join(args.path, 'Tasks'))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            path = os.path.abspath(os.path.join(directory.decode('utf-8'), filename))
            line = predict(path)
            print("{} {}".format(path, line))
            if args.v:
                # verbose mode
                sol = open(path.replace('Tasks', 'Solutions'), 'r').read()
                if not line == int(sol.strip()):
                    incorrect += 1
                    print("********************************************")
                    print("Wrong!")
                    print("Your answer: {}, Solution: {}".format(line, sol))
                    print("Current Recall: {}/{} = {}".format(correct, incorrect + correct, correct / float(incorrect + correct)))
                    print("********************************************")
                else:
                    correct += 1
