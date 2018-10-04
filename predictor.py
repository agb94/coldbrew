import os
import argparse
import difflib
import multiprocessing as mp
from pygments.lexers import get_lexer_by_name
from pygments.token import Keyword, Text

MULTIPROCESSING = True

class JavaKeywords:
    lexer = get_lexer_by_name('java')
    selected_keywords = [
        'try', 'case', 'enum', 'switch', 'catch', 'package',
        'import', 'class', 'for', 'interface', 'void', 'else',
        'static', 'return', 'if', 'long', 'while', 'native']

    @classmethod
    def get(cls, source):
        keywords = []
        for token_type, token in cls.lexer.get_tokens(source):
            if token_type in Keyword and token in cls.selected_keywords:
                keywords.append(token)
        return keywords

class Feature:
    """
    * GP-learned scoring model: div(MBS, neg(mul(HD, div(LS, MBS))))
    
    - MBS: The Sum of Matching Blocks' Size
    - ED: Edit Distance
    - LS: Lexical Similarity
    - KS: Keyword Similarity
    """
    KS_MAGNIFIER = 1.5 # hyperparameter [0.5, 1, 1.5, 2, 2.5, 3]

    def __init__(self, ls, ed, mbs, ks):
        self.ls = float(ls)
        self.ed = float(ed)
        self.mbs = float(mbs)
        self.ks = float(ks)

    def __str__(self):
        return "{:6.3f}l {:6.3f}h {:6.3f}m {:6.3f}k".format(self.ls, self.ed, self.mbs, self.ks)

    def get_score(self):
        from math import sin, cos
        from operator import add, sub, mul, neg
        div = lambda a, b: a / b if b != 0 else 1 # protected division operator
        return div(self.mbs, neg(mul(self.ed, div(self.ls, self.mbs)))) + self.ks * Feature.KS_MAGNIFIER

    @classmethod
    def new(cls, ingredient, line, ing_keywords, line_keywords):
        ks = 1 - difflib.SequenceMatcher(None, ing_keywords, line_keywords).ratio()
        seq_matcher = difflib.SequenceMatcher(lambda x: x == " ", ingredient, line)
        ls = 1 - seq_matcher.ratio()
        ed = len(list(filter(lambda op: op[0] != 'equal', seq_matcher.get_opcodes())))
        mbs = sum(map(lambda b: b.size, seq_matcher.get_matching_blocks()))
        return cls(ls, ed, mbs, ks)

"""
Function Wrappers for Multiprocessing
"""
#====================================
def calculate_score(feature):
    return feature.get_score()

def get_feature(args):
    return Feature.new(*args)
#====================================

def extract_unit(ingredient, source, pool):
    features = list()
    ing_keywords = JavaKeywords.get(ingredient)

    if pool:
        features = pool.map(get_feature, list(map(lambda line: (ingredient, line, ing_keywords, JavaKeywords.get(line)), source)))
    else:
        features = list(map(lambda line: get_feature((ingredient, line, ing_keywords, JavaKeywords.get(line))), source))

    max_mbs = float(max([f.mbs for f in features]))
    for feature in features:
        feature.mbs /= max_mbs

    return features

def predict_unit(ingredient: str, source: list, pool, verbose=False):
    assert type(ingredient) == str
    assert type(source) == list and all(type(line) == str for line in source)

    features = extract_unit(ingredient, source, pool)
    if verbose:
        # verbose mode
        values = [feature.get_score() for feature in features]
        print("===========================================")
        print(ingredient)
        for i, lineno in enumerate(sorted(range(len(values)), key=lambda k: values[k])[:20]):
            print("{}\t{:.6f}\t{}\t{:>5}:\t{}".format(i+1, values[lineno], features[lineno], lineno + 1, source[lineno]))

    if pool:
        scores = pool.map(calculate_score, features)
    else:
        scores = list(map(lambda f: calculate_score(f), features))

    min_value = min(scores)
    solution = -1
    for i in reversed(range(len(scores))):
        if scores[i] == min_value:
            solution = i + 1
            break

    return solution

def predict(path, pool, verbose=False):
    with open(path, 'r') as f:
        lines = f.readlines()
        ingredient = lines[0].strip()
        source = list(map(lambda l: l.strip(), lines[2:]))
        return predict_unit(ingredient, source, pool, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()

    if args.v:
        correct, incorrect = 0, 0

    pool = mp.Pool(mp.cpu_count() - 1) if MULTIPROCESSING else None

    directory = os.fsencode(os.path.join(args.path, 'Tasks'))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            path = os.path.abspath(os.path.join(directory.decode('utf-8'), filename))
            line = predict(path, pool, verbose=args.v)
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
