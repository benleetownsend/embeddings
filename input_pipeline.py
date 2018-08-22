import tqdm
import itertools
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import random
import glob


import tensorflow as tf


class InputPipeline:
    def __init__(self, train_filename=None, val_filename=None, test_filename=None, vocab_size=5000,
                 eos="<EOS>", bos="<BOS>", unk="<UNK>", pad="<pad>", tokens_per_batch=1000,
                 max_lengths_per_bucket=None, allow_smaller_final_batch=True):
        self.tokenizer = WordPunctTokenizer()
        self.eos = eos
        self.bos = bos
        self.unk = unk
        self.pad = pad
        self.special_tokens = [eos, bos, unk, pad]
        self.target_vocab_size = vocab_size
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.test_filename = test_filename
        self.vocab_ = None
        self.unk_id = None
        self.pad_id = None
        self.tokens_per_batch = tokens_per_batch
        self.max_lengths_per_bucket = max_lengths_per_bucket or [15, 50, 100, 1000]
        self.allow_smaller_final_batch = allow_smaller_final_batch

    def tokenize(self, string: str) -> list:
        return [self.bos] + self.tokenizer.tokenize(string.lower()) + [self.eos]

    def token_generator(self, filename_glob: str) -> iter:
        filenames = glob.glob(filename_glob)
        random.shuffle(filenames)
        for filename in filenames:
            with open(filename, "rt") as fp:
                for line in fp:
                    yield self.tokenize(line.strip())

    def build_vocabulary(self, token_gen: iter, size) -> list:
        token_gen = tqdm.tqdm(itertools.chain.from_iterable(token_gen))
        count = Counter(token_gen)
        vocab = [word[0] for word in count.most_common(size - len(self.special_tokens))]
        [vocab.remove(a) for a in self.special_tokens if a in vocab]
        vocab = self.special_tokens + vocab
        return vocab

    @property
    def vocab(self) -> dict:
        if self.vocab_ is None:
            size = self.target_vocab_size
            if "*" in self.train_filename:
                root_path = self.train_filename.split("*")[0] + "data"
            else:
                root_path = self.train_filename.rsplit(".", 1)[0]
            vocab_filename = root_path + ".vocab_{}".format(size)

            try:
                with open(vocab_filename, "rt") as fp:
                    tf.logging.info("Loading vocabulary...")
                    self.vocab_ = dict(
                        zip(map(lambda x: x.strip(), tqdm.tqdm(fp)), range(size))
                    )
            except IOError:
                tf.logging.info("Building vocabulary..")
                vocab = self.build_vocabulary(self.token_generator(self.train_filename), size=size)
                with open(vocab_filename, "wt") as fp:
                    fp.writelines((v + "\n" for v in vocab))
                self.vocab_ = dict(zip(vocab, range(size)))
            self.unk_id = self.vocab[self.unk]
            self.pad_id = self.vocab[self.pad]

        return self.vocab_

    def token_id_gen(self, filename: str) -> iter:
        for line in self.token_generator(filename):
            yield [self.vocab.get(word, self.unk_id) for word in line]

    def bucket_to_sizes(self, input_generator):
        batch_size = [self.tokens_per_batch // x for x in self.max_lengths_per_bucket]
        buckets = [[] for _ in range(len(self.max_lengths_per_bucket))]
        lengths = [[] for _ in range(len(self.max_lengths_per_bucket))]
        for item in input_generator:
            for i, ml in enumerate(self.max_lengths_per_bucket):
                if len(item) <= ml:
                    break
            buckets[i].append((item + ml * [self.pad_id])[:ml])
            lengths[i].append(len(item))
            if len(buckets[i]) >= batch_size[i]:
                yield {
                    "seqs": np.asarray(buckets[i], dtype=np.int),
                    "lens": np.asarray(lengths[i], dtype=np.int)
                }
                buckets[i], lengths[i] = [], []
        if self.allow_smaller_final_batch:
            for buck, leng in zip(buckets, lengths):
                if buck:
                    yield {
                        "seqs": np.asarray(buck, dtype=np.int),
                        "lens": np.asarray(leng, dtype=np.int)
                    }

    def get_input_fn(self, key="train", buffer_size=1000):
        def input_fn():
            assert key in ["train", "test", "val"]
            filename = getattr(self, "{}_filename".format(key))
            assert filename is not None
            dataset = tf.data.Dataset.from_generator(
                lambda: self.bucket_to_sizes(self.token_id_gen(filename)),
                {
                    "seqs": tf.int32,
                    "lens": tf.int32
                },
                {
                    "seqs": tf.TensorShape([None, None]),
                    "lens": tf.TensorShape([None])
                }
            )
            return dataset.shuffle(buffer_size).repeat().prefetch(buffer_size=buffer_size).make_one_shot_iterator().get_next()
        return input_fn

    @property
    def vocab_size(self):
        return len(self.vocab)

