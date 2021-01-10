import numpy as np


def readfile(f):
    labels = []
    for line in open(f):
        line = line.strip()
        if line != '':
            labels.append(line)
    return labels


labels = readfile('labels.txt')
examples = readfile('examples.txt')

examples = np.array(examples).reshape((-1, 3)).tolist()

print(examples)
print(labels)
