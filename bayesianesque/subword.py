from collections import Counter
import tqdm


def n_grams(tokens, n):
    return ("".join(a) for a in zip(*[tokens[i:] for i in range(n)]))


class SubstringTokenizer:
    def __init__(self):
        self.vocab = []  # sorted from longest to shortest.
        self.vocab_hash = None

    def init_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_hash = hash(frozenset(self.vocab))

    def tokenize(self, string):
        if len(string) == 0 or self.vocab is None:
            return list(string)
        for v in self.vocab:
            if v in string:
                splits = string.split(v)
                token_parts = [self.tokenize(s) for s in splits]
                output = []
                for t_p in token_parts[:-1]:
                    output.extend(t_p)
                    output.append(v)
                output.extend(token_parts[-1])
                return output
        return list(string)

    def build(self, generator_fn, vocab_size, min_count, num_itters=10, max_ngrams=3):
        self.counter = Counter()
        generator_fn_with_tqdm = lambda: tqdm.tqdm(generator_fn())
        for i in range(num_itters):

            print("Its iter {}".format(i))
            for sequence in generator_fn_with_tqdm():
                tokens = self.tokenize(sequence)
                self.counter.update(tokens)
                for i in range(2, max_ngrams + 1):
                    self.counter.update(n_grams(tokens, i))

            for token, n in self.counter.items():
                if n > min_count:
                    self.vocab.append(token)

            self.vocab = sorted(self.vocab, key=lambda x: 100000000000000000 - len(x))

            for sequence in generator_fn_with_tqdm():
                tokens = self.tokenize(sequence)
                self.counter.update(tokens)
            self.vocab = []
            for token, n in self.counter.most_common(vocab_size):
                if n > min_count:
                    self.vocab.append(token)
                else:
                    break  # Counter is sorted by most common to least
            self.vocab = sorted(self.vocab, key=lambda x: 100000000000000000 - len(x))
            new_vocab_hash = hash(frozenset(self.vocab))
            if self.vocab_hash == new_vocab_hash:
                print("Algorithm has converged. exiting")
                return
            self.vocab_hash = new_vocab_hash
