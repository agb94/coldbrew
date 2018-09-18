import os
import argparse

MODEL = "div(MBS, neg(mul(HD, div(LS, MBS))))"
FEATURE_NAMES = ['LS', 'HD', 'MBS']
"""
- MBS: the sum of matching blocks' size
- HD: Hamming Distance
- LS: Lexical Similarity
"""

def analyzer(s: str):
    assert type(s) == str
    tokens = list()
    token = ''
    for c in s:
        if not c.isalnum():
            if token:
                tokens.append(curr_token)
                curr_token = ''
            if c != ' ' and not c in [' ', '(', ')', ',']:
                tokens.append(c)
            continue
        token += c
    return tokens

def compute_score(*args):
    from math import sin, cos
    from operator import add, sub, mul, neg
    
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    div = lambda a, b: protectedDiv(a, b)
    
    assert len(args) == len(FEATURE_NAMES)
    model = MODEL
    for i, feature_name in enumerate(FEATURE_NAMES):
        model = model.replace(feature_name, 'args[{}]'.format(i))
    return eval(model)

def extract_unit(ingredient, source):
    import difflib
    features = list()
    max_mbs = 0
    for i in range(len(source)):
        line = source[i]
        seq_matcher = difflib.SequenceMatcher(lambda x: x == " ", ingredient, line)
        ls = 1 - seq_matcher.ratio()
        hd = len(list(filter(lambda op: op[0] != 'equal', seq_matcher.get_opcodes())))
        mbs = sum(map(lambda b: b.size, seq_matcher.get_matching_blocks()))
        max_mbs = max(mbs, max_mbs)
        features.append([ls, hd, mbs])

    for row in features:
        row[-1] /= float(max_mbs)
    
    return features

def predict_unit(ingredient: str, source: list, verbose=False):
    assert type(ingredient) == str
    assert type(source) == list and all(type(line) == str for line in source)

    features = extract_unit(ingredient, source)
    values = [compute_score(*row) for row in features]

    if verbose:
        print("===========================================")
        print(ingredient)
        for i, lineno in enumerate(sorted(range(len(values)), key=lambda k: values[k])[:20]):
            print("{}\t{:.6f}\t{:>5}:\t{}".format(i+1, values[lineno], lineno + 1, source[lineno]))
    min_value = min(values)
    candidates = list()
    for j, value in enumerate(values):
        if value == min_value:
            candidates.append(j)

    # Return the middle one if many candidates exist
    return candidates[len(candidates)//2] + 1

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
