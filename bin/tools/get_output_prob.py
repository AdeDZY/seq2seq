import numpy as np
import argparse
from tensorflow import gfile
from seq2seq.data.vocab import *

def log_prob(tids, logits):
    res = 1
    for t, l in zip(tids, logits):
        p = 1/(1 + np.exp(-l[t]))
        res += np.log(p)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_file")
    parser.add_argument("logits_dump")
    parser.add_argument("target")
    parser.add_argument("output_file")
    args = parser.parse_args()

    # read vocab
    with gfile.GFile(args.vocab_file) as file:
        vocab = list(line.strip("\n") for line in file)
    vocab_size = len(vocab)

    has_counts = len(vocab[0].split("\t")) == 2
    if has_counts:
        vocab, counts = zip(*[_.split("\t") for _ in vocab])
        counts = [float(_) for _ in counts]
        vocab = list(vocab)
    else:
        counts = [-1. for _ in vocab]

    # Add special vocabulary items
    special_vocab = get_special_vocab(vocab_size)
    vocab += list(special_vocab._fields)
    vocab_size += len(special_vocab)
    counts += [-1. for _ in list(special_vocab._fields)]

    print counts
    default = special_vocab.UNK
    vocab_to_id_map = dict(zip(vocab, counts))

    # read logits
    all_logits = np.load(args.logits_dump)

    # read target
    line_number = 0
    with gfile.GFile(args.target) as file:
        for line in file:
            words = line.strip.split(' ')
            ids = [vocab_to_id_map.get(word, default) for word in words]
            logits = all_logits.get('arr_{0}'.format(line_number), None)
            if not logits:
                print "EMPTY"
            assert len(ids) == len(logits) + 1
            lp = log_prob(ids, logits)
            print lp

