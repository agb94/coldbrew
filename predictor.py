import os
import argparse
from pygments.lexers import get_lexer_by_name
from pygments.token import string_to_tokentype

JL = get_lexer_by_name('java')
TOKENTYPE_KEYWORD = string_to_tokentype('Token.Keyword')

class Feature:
    """
    * SCORING MODEL: div(MBS, neg(mul(HD, div(LS, MBS))))
    
    - MBS: The Sum of Matching Blocks' Size
    - HD: Hamming Distance
    - LS: Lexical Similarity
    """
    def __init__(self, ls, hd, mbs, ks):
        self.ls = float(ls)
        self.hd = float(hd)
        self.mbs = float(mbs)
        self.ks = float(ks)
    
    def get_score(self):
        from math import sin, cos
        from operator import add, sub, mul, neg
        div = lambda a, b: a / b if b != 0 else 1 # protected division operator
        return div(self.mbs, neg(mul(self.hd, div(self.ls, self.mbs))))

def extract_unit(ingredient, source):
    import difflib

    features = list()
    max_mbs = 0

    keywords = []
    for token in JL.get_tokens(ingredient):
        if token[0] == TOKENTYPE_KEYWORD:
            keywords.append(token[1])

    for line in source:

        line_keywords = []
        for token in JL.get_tokens(line):
            if token[0] == TOKENTYPE_KEYWORD:
                line_keywords.append(token[1])
        ks = 1 - difflib.SequenceMatcher(None,keywords,line_keywords).ratio()

        seq_matcher = difflib.SequenceMatcher(lambda x: x == " ", ingredient, line)
        ls = 1 - seq_matcher.ratio()
        hd = len(list(filter(lambda op: op[0] != 'equal', seq_matcher.get_opcodes())))
        mbs = sum(map(lambda b: b.size, seq_matcher.get_matching_blocks()))
        max_mbs = max(mbs, max_mbs)
        features.append(Feature(ls, hd, mbs, ks))

    for feature in features:
        # Normalization
        feature.mbs /= float(max_mbs)
    
    return features

def predict_unit(ingredient: str, source: list, verbose=False):
    assert type(ingredient) == str
    assert type(source) == list and all(type(line) == str for line in source)

    features = extract_unit(ingredient, source)
    if verbose:
        # verbose mode
        values = [feature.get_score() for feature in features]
        print("===========================================")
        print(ingredient)
        for i, lineno in enumerate(sorted(range(len(values)), key=lambda k: values[k])[:20]):
            print("{}\t{:.6f}\t{:>5}:\t{}".format(i+1, values[lineno], lineno + 1, source[lineno]))

    min_value = features[0].get_score()
    candidates = list()
    for i, feature in enumerate(features):
        score = feature.get_score()
        if score > min_value:
            pass
        elif score < min_value:
            min_value = score
            candidates = [i]
        else:
            candidates.append(i)

    # 1. Return the middle one if many candidates exist
    # return candidates[len(candidates)//2] + 1

    # 2. Return the last one if many candidates exist
    return candidates[-1] + 1

def predict(path, verbose=False):
    with open(path, 'r') as f:
        lines = f.readlines()
        ingredient = lines[0].strip()
        source = list(map(lambda l: l.strip(), lines[2:]))
        return predict_unit(ingredient, source, verbose=verbose)

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
            line = predict(path, verbose=args.v)
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
