from math import log
from nltk.corpus import brown
from collections import Counter


def brown_sentence_iterator():
    for doc_id in brown.fileids():
        sentences = brown.sents(doc_id)
        for sent in sentences:
            yield sent


def brown_word_iterator():
    sit = brown_sentence_iterator()
    for sent in sit:
        for word in sent:
            yield word


def brown_ngram_iterator(order):
    if order < 2:
        raise ValueError("order must be at least 2")

    for sent in brown_sentence_iterator():
        for idx in range(len(sent) - (order - 1)):
            yield tuple(sent[idx:idx + order])


def calc_total_number_of_words():
    sit = brown_sentence_iterator()
    n_words = 0
    for s in sit:
        n_words += len(s)
    return n_words


def make_brown_vocab():
    wit = brown_word_iterator()
    vocab = set()
    for word in wit:
        vocab.add(word)
    return vocab


class BrownNgramModel(object):

    def __init__(self, order=2):
        self._order = order
        self._ngram_counters = dict()
        self._count_ngrams()

    def _count_ngrams(self):
        self._ngram_counters[1] = Counter(brown_word_iterator())
        for n in range(1, self._order):
            self._ngram_counters[n+1] = Counter(brown_ngram_iterator(n+1))

    @property
    def order(self):
        return self._order

    @property
    def unigram_counts(self):
        return self.get_ngram_counts(1)

    def get_ngram_counts(self, order):
        return self._ngram_counters[order]

    @property
    def vocab(self):
        return self.unigram_counts.keys()

    def bigram_log_prob(self, bigram):
        assert len(bigram)==2
        bigram = bigram if isinstance(bigram, tuple) else tuple(bigram)
        logprob = log(self.get_ngram_counts(2)[bigram] + 1) - log(self.unigram_counts[bigram[0]] + self.vocab_size)
        return logprob

    # def ngram_log_prob(self, ngram):
    #     order = len(ngram)
    #     ngram = ngram if isinstance(ngram, tuple) else tuple(ngram)
    #
    #     logprob = log(self.get_ngram_counts(order)[ngram])

    # @property
    # def total_nwords(self):

    @property
    def vocab_size(self):
        return len(self.vocab)

    # def _count_ngrams(self, order):
    #     for sent in brown_sentence_iterator():


def main():
    model = BrownNgramModel()
    uct = model.unigram_counts
    bct = model.get_ngram_counts(2)
    # print(ctr)


if __name__ == '__main__':
    main()
