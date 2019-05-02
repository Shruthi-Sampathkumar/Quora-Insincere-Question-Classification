import argparse
import json
import math
import os
import sys
import unidecode
import random
import re
import time
import yaml
from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nltk
import gensim
import sklearn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, Doc2Vec, FastText
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from scipy.io import savemat

get_ipython().run_line_magic('load_ext', 'Cython')

class ExperimentConfigBuilderBase(metaclass=ABCMeta):

    default_config = None

    def add_args(self, parser):
        parser.add_argument('--modelfile', '-m', type=Path)
        parser.add_argument('--outdir-top', type=Path, default=Path('results'))
        parser.add_argument('--outdir-bottom', type=str, default='default')
        parser.add_argument('--device', '-g', type=int)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--logging', action='store_true')
        parser.add_argument('--n-rows', type=int)

        parser.add_argument('--seed', type=int, default=1029)
        parser.add_argument('--optuna-trials', type=int)
        parser.add_argument('--gridsearch', action='store_true')
        parser.add_argument('--holdout', action='store_true')
        parser.add_argument('--cv', type=int, default=5)
        parser.add_argument('--cv-part', type=int)
        parser.add_argument('--processes', type=int, default=2)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--batchsize', type=int, default=512)
        parser.add_argument('--batchsize-valid', type=int, default=1024)
        parser.add_argument('--scale-batchsize', type=int, nargs='+',
                            default=[])
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--validate-from', type=int)
        parser.add_argument('--pos-weight', type=float, default=1.)
        parser.add_argument('--maxlen', type=float, default=72)
        parser.add_argument('--vocab-mincount', type=float, default=5)
        parser.add_argument('--ensembler-n-snapshots', type=int, default=1)

    @abstractmethod
    def modules(self):
        raise NotImplementedError()

    def build(self, args=None):
        assert self.default_config is not None
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        parser.set_defaults(**self.default_config)

        for module in self.modules:
            module.add_args(parser)
        config, extra_config = parser.parse_known_args(args)

        for module in self.modules:
            if hasattr(module, 'add_extra_args'):
                module.add_extra_args(parser, config)

        if config.test:
            parser.set_defaults(**dict(
                n_rows=500,
                batchsize=64,
                validate_from=0,
                epochs=3,
                cv_part=2,
                ensembler_test_size=1.,
            ))

        config = parser.parse_args(args)
        if config.modelfile is not None:
            config.outdir = config.outdir_top / config.modelfile.stem \
                / config.outdir_bottom
        else:
            config.outdir = Path('.')

        return config
		
def build_model(config, embedding_matrix, n_sentence_extra_features):
    embedding = Embedding(config, embedding_matrix)
    encoder = Encoder(config, embedding.out_size)
    aggregator = Aggregator(config)
    mlp = MLP(config, encoder.out_size + n_sentence_extra_features)
    out = nn.Linear(config.mlp_n_hiddens[-1], 1)
    lossfunc = nn.BCEWithLogitsLoss()

    return BinaryClassifier(
        embedding=embedding,
        encoder=encoder,
        aggregator=aggregator,
        mlp=mlp,
        out=out,
        lossfunc=lossfunc,
    )

def load_qiqc(n_rows=None):
    #train_df = pd.read_csv(f'./data/train.csv', nrows=n_rows)
    #submit_df = pd.read_csv(f'./data/test.csv', nrows=n_rows)
    train_df=pd.read_csv("train.csv", nrows=n_rows)
    submit_df = pd.read_csv("test.csv",nrows=n_rows)
    n_labels = {
        0: (train_df.target == 0).sum(),
        1: (train_df.target == 1).sum(),
    }
    train_df['target'] = train_df.target.astype('f')
    train_df['weights'] = train_df.target.apply(lambda t: 1 / n_labels[t])

    return train_df, submit_df


def build_datasets(train_df, submit_df, holdout=False, seed=0):
    submit_dataset = QIQCDataset(submit_df)
    if holdout:
        # Train : Test split for holdout training
        splitter = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed)
        train_indices, test_indices = list(splitter.split(
            train_df, train_df.target))[0]
        train_indices.sort(), test_indices.sort()
        train_dataset = QIQCDataset(
            train_df.iloc[train_indices].reset_index(drop=True))
        test_dataset = QIQCDataset(
            train_df.iloc[test_indices].reset_index(drop=True))
    else:
        train_dataset = QIQCDataset(train_df)
        test_dataset = QIQCDataset(train_df.head(0))

    return train_dataset, test_dataset, submit_dataset


class QIQCDataset(object):

    def __init__(self, df):
        self.df = df

    @property
    def tokens(self):
        return self.df.tokens.values

    @tokens.setter
    def tokens(self, tokens):
        self.df['tokens'] = tokens

    @property
    def positives(self):
        return self.df[self.df.target == 1]

    @property
    def negatives(self):
        return self.df[self.df.target == 0]

    def build(self, device):
        self._X = self.tids
        self.X = torch.Tensor(self._X).type(torch.long).to(device)
        if 'target' in self.df:
            self._t = self.df.target[:, None]
            self._W = self.df.weights
            self.t = torch.Tensor(self._t).type(torch.float).to(device)
            self.W = torch.Tensor(self._W).type(torch.float).to(device)
        if hasattr(self, '_X2'):
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)
        else:
            self._X2 = np.zeros((self._X.shape[0], 1), 'f')
            self.X2 = torch.Tensor(self._X2).type(torch.float).to(device)

    def build_labeled_dataset(self, indices):
        return torch.utils.data.TensorDataset(
            self.X[indices], self.X2[indices],
            self.t[indices], self.W[indices])

# Registries for preprocessing
NORMALIZER_REGISTRY = {}
TOKENIZER_REGISTRY = {}
WORD_EMBEDDING_FEATURIZER_REGISTRY = {}
WORD_EXTRA_FEATURIZER_REGISTRY = {}
SENTENCE_EXTRA_FEATURIZER_REGISTRY = {}

# Registries for training
ENCODER_REGISTRY = {}
AGGREGATOR_REGISTRY = {}
ATTENTION_REGISTRY = {}


def register_preprocessor(name):
    def register_cls(cls):
        NORMALIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_tokenizer(name):
    def register_cls(cls):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_word_embedding_features(name):
    def register_cls(cls):
        WORD_EMBEDDING_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_word_extra_features(name):
    def register_cls(cls):
        WORD_EXTRA_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_sentence_extra_features(name):
    def register_cls(cls):
        SENTENCE_EXTRA_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_encoder(name):
    def register_cls(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_aggregator(name):
    def register_cls(cls):
        AGGREGATOR_REGISTRY[name] = cls
        return cls
    return register_cls


def register_attention(name):
    def register_cls(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return register_cls

class WordVocab(object):

    def __init__(self, mincount=1):
        self.counter = Counter()
        self.n_documents = 0
        self._counters = {}
        self._n_documents = defaultdict(int)
        self.mincount = mincount
#         self.unk =  vocab.unk.astype('f')

    def __len__(self):
        return len(self.token2id)

    def add_documents(self, documents, name):
        self._counters[name] = Counter()
        for document in documents:
            bow = dict.fromkeys(document, 1)
            self._counters[name].update(bow)
            self.counter.update(bow)
            self.n_documents += 1
            self._n_documents[name] += 1

    def build(self):
        counter = dict(self.counter.most_common())
        self.word_freq = {
            **{'<PAD>': 0},
            **counter,
        }
        self.token2id = {
            **{'<PAD>': 0},
            **{word: i + 1 for i, word in enumerate(counter)}
        }
        self.lfq = np.array(list(self.word_freq.values())) < self.mincount
        self.hfq = ~self.lfq

#######################################  %%cython

get_ipython().run_cell_magic('cython', '', 'import re\n\nimport numpy as np\ncimport numpy as np\n\n\ncdef class StringReplacer:\n    cpdef public dict rule\n    cpdef list keys\n    cpdef list values\n    cpdef int n_rules\n\n    def __init__(self, dict rule):\n        self.rule = rule\n        self.keys = list(rule.keys())\n        self.values = list(rule.values())\n        self.n_rules = len(rule)\n\n    def __call__(self, str x):\n        cdef int i\n        for i in range(self.n_rules):\n            if self.keys[i] in x:\n                x = x.replace(self.keys[i], self.values[i])\n        return x\n\n    def __getstate__(self):\n        return (self.rule, self.keys, self.values, self.n_rules)\n\n    def __setstate__(self, state):\n        self.rule, self.keys, self.values, self.n_rules = state\n\n\ncdef class RegExpReplacer:\n    cdef dict rule\n    cdef list keys\n    cdef list values\n    cdef regexp\n    cdef int n_rules\n\n    def __init__(self, dict rule):\n        self.rule = rule\n        self.keys = list(rule.keys())\n        self.values = list(rule.values())\n        self.regexp = re.compile(\'(%s)\' % \'|\'.join(self.keys))\n        self.n_rules = len(rule)\n\n    @property\n    def rule(self):\n        return self.rule\n\n    def __call__(self, str x):\n        def replace(match):\n            x = match.group(0)\n            if x in self.rule:\n                return self.rule[x]\n            else:\n                for i in range(self.n_rules):\n                    x = re.sub(self.keys[i], self.values[i], x)\n                return x\n        return self.regexp.sub(replace, x)\n\n\ncpdef str cylower(str x):\n    return x.lower()\n\n\nCache = {}\nis_alphabet = re.compile(r\'[a-zA-Z]\')\n\n\ncpdef str unidecode_weak(str string):\n    """Transliterate an Unicode object into an ASCII string\n    >>> unidecode(u"\\u5317\\u4EB0")\n    "Bei Jing "\n    """\n\n    cdef list retval = []\n    cdef int i = 0\n    cdef int n = len(string)\n    cdef str char\n\n    for i in range(n):\n        char = string[i]\n        codepoint = ord(char)\n\n        if codepoint < 0x80: # Basic ASCII\n            retval.append(char)\n            continue\n\n        if codepoint > 0xeffff:\n            continue  # Characters in Private Use Area and above are ignored\n\n        section = codepoint >> 8   # Chop off the last two hex digits\n        position = codepoint % 256 # Last two hex digits\n\n        try:\n            table = Cache[section]\n        except KeyError:\n            try:\n                mod = __import__(\'unidecode.x%03x\'%(section), [], [], [\'data\'])\n            except ImportError:\n                Cache[section] = None\n                continue   # No match: ignore this character and carry on.\n\n            Cache[section] = table = mod.data\n\n        if table and len(table) > position:\n            if table[position] == \'[?]\' or is_alphabet.match(table[position]):\n                retval.append(\' \' + char + \' \')\n            else:\n                retval.append(table[position])\n\n    return \'\'.join(retval)')



class PunctSpacer(StringReplacer):

    def __init__(self, edge_only=False):
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', '█', '½', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]  # NOQA
        if edge_only:
            rule = {
                **dict([(f' {p}', f' {p} ') for p in puncts]),
                **dict([(f'{p} ', f' {p} ') for p in puncts]),
            }
        else:
            rule = dict([(p, f' {p} ') for p in puncts])
        super().__init__(rule)


class NumberReplacer(RegExpReplacer):

    def __init__(self, with_underscore=False):
        prefix, suffix = '', ''
        if with_underscore:
            prefix += ' __'
            suffix = '__ '
        rule = {
            '[0-9]{5,}': f'{prefix}#####{suffix}',
            '[0-9]{4}': f'{prefix}####{suffix}',
            '[0-9]{3}': f'{prefix}###{suffix}',
            '[0-9]{2}': f'{prefix}##{suffix}',
        }
        super().__init__(rule)


class KerasFilterReplacer(StringReplacer):

    def __init__(self):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        rule = dict([(f, ' ') for f in filters])
        super().__init__(rule)


class MisspellReplacer(StringReplacer):

    def __init__(self):
        rule = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "how'd'y": "how do you",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd've": "i would have",
            "i'd": "i would",
            "i'll've": "i will have",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd've": "it would have",
            "it'd": "it would",
            "it'll've": "it will have",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't've": "might not have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't've": "must not have",
            "mustn't": "must not",
            "needn't've": "need not have",
            "needn't": "need not",
            "o'clock": "of the clock",
            "oughtn't've": "ought not have",
            "oughtn't": "ought not",
            "shan't've": "shall not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd've": "she would have",
            "she'd": "she would",
            "she'll've": "she will have",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't've": "should not have",
            "shouldn't": "should not",
            "so've": "so have",
            "so's": "so as",
            "this's": "this is",
            "that'd've": "that would have",
            "that'd": "that would",
            "that's": "that is",
            "there'd've": "there would have",
            "there'd": "there would",
            "there's": "there is",
            "here's": "here is",
            "they'd've": "they would have",
            "they'd": "they would",
            "they'll've": "they will have",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd've": "we would have",
            "we'd": "we would",
            "we'll've": "we will have",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll've": "what will have",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll've": "who will have",
            "who'll": "who will",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't've": "will not have",
            "won't": "will not",
            "would've": "would have",
            "wouldn't've": "would not have",
            "wouldn't": "would not",
            "y'all'd've": "you all would have",
            "y'all'd": "you all would",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "y'all": "you all",
            "you'd've": "you would have",
            "you'd": "you would",
            "you'll've": "you will have",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
            "colour": "color",
            "centre": "center",
            "favourite": "favorite",
            "travelling": "traveling",
            "counselling": "counseling",
            "theatre": "theater",
            "cancelled": "canceled",
            "labour": "labor",
            "organisation": "organization",
            "wwii": "world war 2",
            "citicise": "criticize",
            "youtu ": "youtube ",
            "qoura": "quora",
            "sallary": "salary",
            "whta": "what",
            "narcisist": "narcissist",
            "howdo": "how do",
            "whatare": "what are",
            "howcan": "how can",
            "howmuch": "how much",
            "howmany": "how many",
            "whydo": "why do",
            "doi": "do i",
            "thebest": "the best",
            "howdoes": "how does",
            "mastrubation": "masturbation",
            "mastrubate": "masturbate",
            "mastrubating": "masturbating",
            "pennis": "penis",
            "etherium": "ethereum",
            "narcissit": "narcissist",
            "bigdata": "big data",
            "2k17": "2017",
            "2k18": "2018",
            "qouta": "quota",
            "exboyfriend": "ex boyfriend",
            "airhostess": "air hostess",
            "whst": "what",
            "watsapp": "whatsapp",
            "demonitisation": "demonetization",
            "demonitization": "demonetization",
            "demonetisation": "demonetization",
        }
        super().__init__(rule)


register_preprocessor('lower')(cylower)
register_preprocessor('punct')(PunctSpacer())
register_preprocessor('unidecode')(unidecode)
register_preprocessor('unidecode_weak')(unidecode_weak)
register_preprocessor('number')(NumberReplacer())
register_preprocessor('number+underscore')(
    NumberReplacer(with_underscore=True))
register_preprocessor('misspell')(MisspellReplacer())
register_preprocessor('keras')(KerasFilterReplacer())

def load_pretrained_vectors(names, token2id, test=False):
    assert isinstance(names, list)
    with Pool(processes=len(names)) as pool:
        f = partial(load_pretrained_vector, token2id=token2id, test=test)
        vectors = pool.map(f, names)
    return dict([(n, v) for n, v in zip(names, vectors)])


def load_pretrained_vector(name, token2id, test=False):
    loader = dict(
        gnews=GNewsPretrainedVector,
        wnews=WNewsPretrainedVector,
        paragram=ParagramPretrainedVector,
        glove=GlovePretrainedVector,
    )
    return loader[name].load(token2id, test)


class BasePretrainedVector(object):

    @classmethod
    def load(cls, token2id, test=False, limit=None):
        embed_shape = (len(token2id), 300)
        freqs = np.zeros((len(token2id)), dtype='f')

        if test:
            np.random.seed(0)
            vectors = np.random.normal(0, 1, embed_shape)
            vectors[0] = 0
            vectors[len(token2id) // 2:] = 0
        else:
            vectors = np.zeros(embed_shape, dtype='f')
            path = f'{os.environ["DATADIR"]}/{cls.path}'
            for i, o in enumerate(
                    open(path, encoding="utf8", errors='ignore')):
                token, *vector = o.split(' ')
                token = str.lower(token)
                if token not in token2id or len(o) <= 100:
                    continue
                if limit is not None and i > limit:
                    break
                freqs[token2id[token]] += 1
                vectors[token2id[token]] += np.array(vector, 'f')

        vectors[freqs != 0] /= freqs[freqs != 0][:, None]
        vec = KeyedVectors(300)
        vec.add(list(token2id.keys()), vectors, replace=True)

        return vec


class GNewsPretrainedVector(object):

    name = 'GoogleNews-vectors-negative300'
    path = f'{name}/{name}.bin'

    @classmethod
    def load(cls, tokens, limit=None):
        raise NotImplementedError
        path = f'./data/{cls.path}'
        return KeyedVectors.load_word2vec_format(
            path, binary=True, limit=limit)


class WNewsPretrainedVector(BasePretrainedVector):

    name = 'wiki-news-300d-1M'
    path = f'{name}/{name}.vec'


class ParagramPretrainedVector(BasePretrainedVector):

    name = 'paragram_300_sl999'
    path = f'{name}/{name}.txt'


class GlovePretrainedVector(BasePretrainedVector):

    name = 'glove.840B.300d'
    path = f'{name}/{name}.txt'


@register_word_embedding_features('pretrained')
class PretrainedVectorFeaturizer(object):

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    @classmethod
    def add_args(self, parser):
        pass

    def __call__(self, features, datasets):
        # Nothing to do
        return features


class Any2VecFeaturizer(object):

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    def build_fillvalue(self, mode, initialW):
        n_embed = initialW.shape[1]
        n_fill = initialW[self.vocab.unk].shape
        assert mode in {'zeros', 'mean', 'noise'}
        if mode == 'zeros':
            return np.zeros(n_embed, 'f')
        elif mode == 'mean':
            return initialW.mean(axis=0)
        elif mode == 'noise':
            mean, std = initialW.mean(), initialW.std()
            return np.random.normal(mean, std, (n_fill, n_embed))

    def __call__(self, features, datasets):
        tokens = np.concatenate([d.tokens for d in datasets])
        model = self.build_model()
        model.build_vocab_from_freq(self.vocab.word_freq)
        initialW = features.copy()
        initialW[self.vocab.unk] = self.build_fillvalue(
            self.config.finetune_word2vec_init_unk, initialW)
        idxmap = np.array(
            [self.vocab.token2id[w] for w in model.wv.index2entity])
        model = self.initialize(model, initialW, idxmap)
        model.train(tokens, total_examples=len(tokens), epochs=model.epochs)
        finetunedW = np.zeros((initialW.shape), 'f')
        for i, word in enumerate(self.vocab.token2id):
            if word in model.wv:
                finetunedW[i] = model.wv.get_vector(word)
        return finetunedW


@register_word_embedding_features('word2vec')
class Word2VecFeaturizer(Any2VecFeaturizer):

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--finetune-word2vec-init-unk', type=str,
                            choices=['zeros', 'mean', 'noise'])
        parser.add_argument('--finetune-word2vec-mincount', type=int)
        parser.add_argument('--finetune-word2vec-workers', type=int)
        parser.add_argument('--finetune-word2vec-iter', type=int)
        parser.add_argument('--finetune-word2vec-size', type=int)
        parser.add_argument('--finetune-word2vec-window', type=int, default=5)
        parser.add_argument('--finetune-word2vec-sorted-vocab', type=int,
                            default=0)
        parser.add_argument('--finetune-word2vec-sg', type=int, choices=[0, 1])

    def build_model(self):
        model = Word2Vec(
            min_count=self.config.finetune_word2vec_mincount,
            workers=self.config.finetune_word2vec_workers,
            iter=self.config.finetune_word2vec_iter,
            size=self.config.finetune_word2vec_size,
            window=self.config.finetune_word2vec_window,
            sg=self.config.finetune_word2vec_sg,
        )
        return model

    def initialize(self, model, initialW, idxmap):
        model.wv.vectors[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        return model


@register_word_embedding_features('fasttext')
class FastTextFeaturizer(Any2VecFeaturizer):

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--finetune-fasttext-init-unk', type=str,
                            choices=['zeros', 'mean', 'noise'])
        parser.add_argument('--finetune-fasttext-mincount', type=int)
        parser.add_argument('--finetune-fasttext-workers', type=int)
        parser.add_argument('--finetune-fasttext-iter', type=int)
        parser.add_argument('--finetune-fasttext-size', type=int)
        parser.add_argument('--finetune-fasttext-sg', type=int, choices=[0, 1])
        parser.add_argument('--finetune-fasttext-min_n', type=int)
        parser.add_argument('--finetune-fasttext-max_n', type=int)

    def build_model(self):
        model = FastText(
            min_count=self.config.finetune_fasttext_mincount,
            workers=self.config.finetune_fasttext_workers,
            iter=self.config.finetune_fasttext_iter,
            size=self.config.finetune_fasttext_size,
            sg=self.config.finetune_fasttext_sg,
            min_n=self.config.finetune_fasttext_min_n,
            max_n=self.config.finetune_fasttext_max_n,
        )
        return model

    def initialize(self, model, initialW, idxmap):
        model.wv.vectors[:] = initialW[idxmap]
        model.wv.vectors_vocab[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        return model

@register_word_extra_features('idf')
class IDFWordFeaturizer(object):

    def __call__(self, vocab):
        dfs = np.array(list(vocab.word_freq.values()))
        dfs[0] = vocab.n_documents
        features = np.log(vocab.n_documents / dfs)
        features = features[:, None]
        return features

@register_sentence_extra_features('char')
class CharacterStatisticsFeaturizer(object):

    n_dims = 3

    def __call__(self, sentence):
        feature = {}
        feature['n_chars'] = len(sentence)
        feature['n_caps'] = sum(1 for char in sentence if char.isupper())
        feature['caps_rate'] = feature['n_caps'] / feature['n_chars']
        features = np.array(list(feature.values()))
        return features


@register_sentence_extra_features('word')
class WordStatisticsFeaturizer(object):

    n_dims = 3

    def __call__(self, sentence):
        feature = {}
        tokens = sentence.split()
        feature['n_words'] = len(tokens)
        feature['unique_words'] = len(set(tokens))
        feature['unique_rate'] = feature['unique_words'] / feature['n_words']
        features = np.array(list(feature.values()))
        return features


###############################%%cython
get_ipython().run_cell_magic('cython', '', 'cpdef list cysplit(str x):\n    return x.split()')


register_tokenizer('space')(cysplit)
register_tokenizer('word_tokenize')(nltk.word_tokenize)

class TextNormalizerWrapper(object):

    registry = NORMALIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.normalizers = [self.registry[n] for n in config.normalizers]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--normalizers', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, x):
        for normalizer in self.normalizers:
            x = normalizer(x)
        return x

    
class TextTokenizerWrapper(object):

    registry = TOKENIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.tokenizer = self.registry[config.tokenizer]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--tokenizer', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, x):
        return self.tokenizer(x)

    
class WordEmbeddingFeaturizerWrapper(object):

    registry = WORD_EMBEDDING_FEATURIZER_REGISTRY
    default_config = None
    default_extra_config = None

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        self.featurizers = {
            k: self.registry[k](config, vocab)
            for k in config.word_embedding_features}

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--use-pretrained-vectors', nargs='+',
            choices=['glove', 'paragram', 'wnews', 'gnews'])
        parser.add_argument(
            '--word-embedding-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        assert isinstance(cls.default_extra_config, dict)
        for featurizer in config.word_embedding_features:
            cls.registry[featurizer].add_args(parser)
        parser.set_defaults(**cls.default_extra_config)

    def __call__(self, features, datasets):
        return {k: feat(features, datasets)
                for k, feat in self.featurizers.items()}


class WordExtraFeaturizerWrapper(object):

    registry = WORD_EXTRA_FEATURIZER_REGISTRY
    default_config = None

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        self.featurizers = {
            k: self.registry[k]() for k in config.word_extra_features}

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--word-extra-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, vocab):
        empty = np.empty([len(vocab), 0])
        return np.concatenate([empty, *[
            f(vocab) for f in self.featurizers.values()]], axis=1)


class SentenceExtraFeaturizerWrapper(object):

    registry = SENTENCE_EXTRA_FEATURIZER_REGISTRY
    default_config = None

    def __init__(self, config):
        self.config = config
        self.featurizers = {
            k: self.registry[k]() for k in config.sentence_extra_features}
        self.n_dims = sum(list(f.n_dims for f in self.featurizers.values()))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--sentence-extra-features', nargs='+', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    def __call__(self, sentence):
        empty = np.empty((0,))
        return np.concatenate([empty, *[
            f(sentence) for f in self.featurizers.values()]], axis=0)

    def fit_standardize(self, features):
        assert features.ndim == 2
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std = np.where(self.std != 0, self.std, 1)
        return (features - self.mean) / self.std

    def standardize(self, features):
        assert hasattr(self, 'mean'), hasattr(self, 'std')
        return (features - self.mean) / self.std

class WordbasedPreprocessor():

    def tokenize(self, datasets, normalizer, tokenizer):
        tokenize = Pipeline(normalizer, tokenizer)
        apply_tokenize = ApplyNdArray(tokenize, processes=2, dtype=object)
        tokens = [apply_tokenize(d.df.question_text.values) for d in datasets]
        return tokens

    def build_vocab(self, datasets, config):
        train_dataset, test_dataset, submit_dataset = datasets
        vocab = WordVocab(mincount=config.vocab_mincount)
        vocab.add_documents(train_dataset.positives.tokens, 'train-pos')
        vocab.add_documents(train_dataset.negatives.tokens, 'train-neg')
        vocab.add_documents(test_dataset.positives.tokens, 'test-pos')
        vocab.add_documents(test_dataset.negatives.tokens, 'test-neg')
        vocab.add_documents(submit_dataset.df.tokens, 'submit')
        vocab.build()
        return vocab

    def build_tokenids(self, datasets, vocab, config):
        token2id = lambda xs: pad_sequence(  # NOQA
            [vocab.token2id[x] for x in xs], config.maxlen)
        apply_token2id = ApplyNdArray(
            token2id, processes=1, dtype='i', dims=(config.maxlen,))
        tokenids = [apply_token2id(d.df.tokens.values) for d in datasets]
        return tokenids

    def build_sentence_features(self, datasets, sentence_extra_featurizer):
        train_dataset, test_dataset, submit_dataset = datasets
        apply_featurize = ApplyNdArray(
            sentence_extra_featurizer, processes=1, dtype='f',
            dims=(sentence_extra_featurizer.n_dims,))
        _X2 = [apply_featurize(d.df.question_text.values) for d in datasets]
        _train_X2, _test_X2, _submit_X2 = _X2
        train_X2 = sentence_extra_featurizer.fit_standardize(_train_X2)
        test_X2 = sentence_extra_featurizer.standardize(_test_X2)
        submit_X2 = sentence_extra_featurizer.standardize(_submit_X2)
        return train_X2, test_X2, submit_X2

    def build_embedding_matrices(self, datasets, word_embedding_featurizer,
                                 vocab, pretrained_vectors):
        pretrained_vectors_merged = np.stack(
            [wv.vectors for wv in pretrained_vectors.values()]).mean(axis=0)
        vocab.unk = (pretrained_vectors_merged == 0).all(axis=1)
        vocab.known = ~vocab.unk
        embedding_matrices = word_embedding_featurizer(
            pretrained_vectors_merged, datasets)
        return embedding_matrices

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))
        embedding = embedding.mean(axis=0)
        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features

class RNNEncoderBase(nn.Module):

    def __init__(self, config, modules, in_size):
        super().__init__()
        rnns = []
        input_size = in_size
        for module in modules:
            rnn = module(
                input_size=input_size,
                hidden_size=config.encoder_n_hidden,
                bidirectional=config.encoder_bidirectional,
                batch_first=True,
            )
            n_direction = int(config.encoder_bidirectional) + 1
            input_size = n_direction * config.encoder_n_hidden
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)
        self.out_size = n_direction * config.encoder_n_hidden

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--encoder-bidirectional', type=bool, default=True)
        parser.add_argument('--encoder-dropout', type=float, default=0.)
        parser.add_argument('--encoder-n-hidden', type=int)
        parser.add_argument('--encoder-n-layers', type=int)
        parser.add_argument('--encoder-aggregator', type=str,
                            choices=AGGREGATOR_REGISTRY)

    def forward(self, input, mask):
        h = input
        for rnn in self.rnns:
            h, _ = rnn(h)
        return h


@register_encoder('lstm')
class LSTMEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        modules = [nn.LSTM] * config.encoder_n_layers
        super().__init__(config, modules, in_size)


@register_encoder('gru')
class GRUEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.GRU] * config.encoder_n_layers
        super().__init__(config, modules, in_size)


@register_encoder('lstmgru')
class LSTMGRUEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.LSTM] * (config.encoder_n_layers - 1) + [nn.GRU]
        super().__init__(config, modules, in_size)


@register_encoder('grulstm')
class GRULSTMEncoder(RNNEncoderBase):

    def __init__(self, config, in_size):
        assert config.encoder_n_layers > 1
        modules = [nn.GRU] * (config.encoder_n_layers - 1) + [nn.LSTM]
        super().__init__(config, modules, in_size)

@register_aggregator('max')
class MaxPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), -np.inf)
        h = hs.max(dim=1)[0]
        return h


@register_aggregator('sum')
class SumPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), 0)
        h = hs.sum(dim=1)
        return h


@register_aggregator('avg')
class AvgPoolingAggregator(nn.Module):

    def __call__(self, hs, mask):
        if mask is not None:
            hs = hs.masked_fill(~mask.unsqueeze(2), 0)
        h = hs.sum(dim=1)
        maxlen = mask.sum(dim=1)
        h /= maxlen[:, None].type(torch.float)
        return h

class BaseEnsembler(metaclass=ABCMeta):

    def __init__(self, config, models, results):
        super().__init__()
        self.config = config
        self.models = models
        self.results = results

    @abstractmethod
    def fit(self, X, t, test_size=0.1):
        pass

    @abstractmethod
    def predict_proba(self, X, X2):
        pass

    def predict(self, X, X2):
        y = self.predict_proba(X, X2)
        return (y > self.threshold).astype('i')

    
class AverageEnsembler(BaseEnsembler):

    def __init__(self, config, models, results):
        self.config = config
        self.models = models
        self.results = results
        self.device = config.device
        self.batchsize_train = config.batchsize
        self.batchsize_valid = config.batchsize_valid
        self.threshold_cv = np.array(
            [m.threshold for m in models]).mean()
        self.threshold = self.threshold_cv

    def fit(self, X, X2, t, test_size=0.1):
        # Nothing to do
        pass

    def predict_proba(self, X, X2):
        pred_X = X.to(self.device)
        pred_X2 = X2.to(self.device)
        dataset = torch.utils.data.TensorDataset(pred_X, pred_X2)
        iterator = DataLoader(
            dataset, batch_size=self.batchsize_valid, shuffle=False)
        ys = defaultdict(list)
        for batch in tqdm(iterator, desc='submit', leave=False):
            for i, model in enumerate(self.models):
                model.eval()
                ys[i].append(model.predict_proba(*batch))
        ys = np.concatenate(
            [np.concatenate(_ys) for _ys in ys.values()], axis=1)
        y = ys.mean(axis=1, keepdims=True)
        return y
		
class BinaryClassifier(nn.Module):

    default_config = None

    def __init__(self, embedding, encoder, aggregator, mlp, out, lossfunc):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.aggregator = aggregator
        self.mlp = mlp
        self.out = out
        self.lossfunc = lossfunc

    def calc_loss(self, X, X2, t, W=None):
        y = self.forward(X, X2)
        loss = self.lossfunc(y, t)
        output = dict(
            y=torch.sigmoid(y).cpu().detach().numpy(),
            t=t.cpu().detach().numpy(),
            loss=loss.cpu().detach().numpy(),
        )
        return loss, output

    def to_device(self, device):
        self.device = device
        self.to(device)
        return self

    def forward(self, X, X2):
        h = self.predict_features(X, X2)
        out = self.out(h)
        return out

    def predict_proba(self, X, X2):
        y = self.forward(X, X2)
        proba = torch.sigmoid(y).cpu().detach().numpy()
        return proba

    def predict_features(self, X, X2):
        mask = X != 0
        maxlen = (mask == 1).any(dim=0).sum()
        X = X[:, :maxlen]
        mask = mask[:, :maxlen]

        h = self.embedding(X)
        h = self.encoder(h, mask)
        h = self.aggregator(h, mask)
        h = self.mlp(h, X2)
        return h

class NNModuleWrapperBase(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def add_args(cls, parser):
        raise NotImplementedError()

    @abstractmethod
    def add_extra_args(cls, parser):
        raise NotImplementedError()

        
class EmbeddingWrapper(NNModuleWrapperBase):

    default_config = None

    def __init__(self, config, embedding_matrix):
        super().__init__()
        self.config = config
        self.module = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_matrix), freeze=True)
        if self.config.embedding_dropout1d > 0:
            self.dropout1d = nn.Dropout(config.embedding_dropout1d)
        if self.config.embedding_dropout2d > 0:
            self.dropout2d = nn.Dropout2d(config.embedding_dropout2d)
        if self.config.embedding_spatial_dropout > 0:
            self.spatial_dropout = nn.Dropout2d(
                config.embedding_spatial_dropout)
        self.out_size = embedding_matrix.shape[1]

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--embedding-dropout1d', type=float, default=0.)
        parser.add_argument('--embedding-dropout2d', type=float, default=0.)
        parser.add_argument('--embedding-spatial-dropout',
                            type=float, default=0.)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X):
        h = self.module(X)
        if self.config.embedding_dropout1d > 0:
            h = self.dropout1d(h)
        if self.config.embedding_dropout2d > 0:
            h = self.dropout2d(h)
        if self.config.embedding_spatial_dropout > 0:
            h = h.permute(0, 2, 1)
            h = self.spatial_dropout(h)
            h = h.permute(0, 2, 1)
        return h

    
class EncoderWrapper(nn.Module):

    registry = ENCODER_REGISTRY

    def __init__(self, config, in_size):
        super().__init__()
        self.config = config
        self.module = self.registry[config.encoder](config, in_size)
        self.out_size = self.module.out_size

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument(
            '--encoder', choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        assert isinstance(cls.default_extra_config, dict)
        cls.registry[config.encoder].add_args(parser)
        parser.set_defaults(**cls.default_extra_config)

    def forward(self, X, mask):
        h = self.module(X, mask)
        return h

    
class AggregatorWrapper(NNModuleWrapperBase):

    default_config = None
    registry = AGGREGATOR_REGISTRY

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = self.registry[config.aggregator]()

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--aggregator',
                            choices=cls.registry)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X, mask):
        h = self.module(X, mask)
        return h

    
class MLPWrapper(NNModuleWrapperBase):

    default_config = None

    def __init__(self, config, in_size):
        super().__init__()
        self.in_size = in_size
        self.config = config
        assert isinstance(config.mlp_n_hiddens, list)
        layers = []
        if config.mlp_bn0:
            layers.append(nn.BatchNorm1d(in_size))
        if config.mlp_dropout0 > 0:
            layers.append(nn.Dropout(config.mlp_dropout0))
        for n_hidden in config.mlp_n_hiddens:
            layers.append(nn.Linear(in_size, n_hidden))
            if config.mlp_actfun is not None:
                layers.append(config.mlp_actfun)
            if config.mlp_bn:
                layers.append(nn.BatchNorm1d(n_hidden))
            if config.mlp_dropout > 0:
                layers.append(nn.Dropout(config.mlp_dropout))
            in_size = n_hidden
        self.layers = nn.Sequential(*layers)

    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--mlp-n-hiddens', type=list)
        parser.add_argument('--mlp-bn', type=bool)
        parser.add_argument('--mlp-bn0', type=bool)
        parser.add_argument('--mlp-dropout', type=float, default=0.)
        parser.add_argument('--mlp-dropout0', type=float, default=0.)
        parser.add_argument('--mlp-actfun', default=0.)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X, X2):
        h = X
        if X.shape[1] + X2.shape[1] == self.in_size:
            h = torch.cat([h, X2], dim=1)
        h = self.layers(h)
        return h

class ExperimentConfigBuilderPresets(ExperimentConfigBuilderBase):

    default_config = dict(
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=2,
    )


# =======  Preprocessing modules  =======

class TextNormalizerPresets(TextNormalizerWrapper):

    default_config = dict(
        normalizers=[
            'lower',
            'misspell',
            'punct',
            'number+underscore'
        ]
    )


class TextTokenizerPresets(TextTokenizerWrapper):

    default_config = dict(
        tokenizer='space'
    )

class WordEmbeddingFeaturizerPresets(WordEmbeddingFeaturizerWrapper):

    default_config = dict(
        use_pretrained_vectors=['glove', 'paragram'],
        word_embedding_features=['pretrained', 'word2vec'],
    )
    default_extra_config = dict(
        finetune_word2vec_init_unk='zeros',
        finetune_word2vec_mincount=1,
        finetune_word2vec_workers=1,
        finetune_word2vec_iter=5,
        finetune_word2vec_size=300,
        finetune_word2vec_sg=0,
        finetune_word2vec_sorted_vocab=0,
    )


class WordExtraFeaturizerPresets(WordExtraFeaturizerWrapper):

    default_config = dict(
        word_extra_features=[],
    )


class SentenceExtraFeaturizerPresets(SentenceExtraFeaturizerWrapper):

    default_config = dict(
        sentence_extra_features=[],
    )


class PreprocessorPresets(WordbasedPreprocessor):

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))

        # Add noise
        unk = (embedding[0] == 0).all(axis=1)
        mean, std = embedding[0, ~unk].mean(), embedding[0, ~unk].std()
        unk_and_hfq = unk & word_embedding_featurizer.vocab.hfq
        noise = np.random.normal(
            mean, std, (unk_and_hfq.sum(), embedding[0].shape[1]))
        embedding[0, unk_and_hfq] = noise
        embedding[0, 0] = 0

        embedding = embedding.mean(axis=0)
        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features


# =======  Training modules  =======

class EmbeddingPresets(EmbeddingWrapper):

    default_config = dict(
        embedding_dropout1d=0.2,
    )


class EncoderPresets(EncoderWrapper):

    default_config = dict(
        encoder='lstm',
    )
    default_extra_config = dict(
        encoder_bidirectional=True,
        encoder_dropout=0.,
        encoder_n_layers=2,
        encoder_n_hidden=128,
    )


class AggregatorPresets(AggregatorWrapper):

    default_config = dict(
        aggregator='max',
    )


class MLPPresets(MLPWrapper):

    default_config = dict(
        mlp_n_hiddens=[128, 128],
        mlp_bn0=False,
        mlp_dropout0=0.,
        mlp_bn=True,
        mlp_actfun=nn.ReLU(True),
    )


class EnsemblerPresets(AverageEnsembler):
    pass

def classification_metrics(ys, ts):
    scores = {}

    if len(np.unique(ts)) > 1:
        # Search optimal threshold
        precs, recs, thresholds = metrics.precision_recall_curve(ts, ys)
        thresholds = np.append(thresholds, 1.001)
        idx = (precs != 0) * (recs != 0)
        precs, recs, thresholds = precs[idx], recs[idx], thresholds[idx]
        fbetas = 2 / (1 / precs + 1 / recs)
        best_idx = np.argmax(fbetas)
        threshold = thresholds[best_idx]
        prec = precs[best_idx]
        rec = recs[best_idx]
        fbeta = fbetas[best_idx]

        scores['ap'] = metrics.average_precision_score(ts, ys)
        scores['rocauc'] = metrics.roc_auc_score(ts, ys)
        scores['threshold'] = threshold
        scores['prec'] = prec
        scores['rec'] = rec
        scores['fbeta'] = fbeta

    return scores


class ClassificationResult(object):

    def __init__(self, name, outdir=None, postfix=None, main_metrics='fbeta'):
        self.initialize()
        self.name = name
        self.postfix = postfix
        self.outdir = outdir
        self.summary = None
        self.main_metrics = main_metrics
        self.n_trained = 0

    def initialize(self):
        self.losses = []
        self.ys = []
        self.ts = []

    def add_record(self, loss, y, t):
        self.losses.append(loss)
        self.ys.append(y)
        self.ts.append(t)
        self.n_trained += len(y)

    def calc_score(self, epoch):
        loss = np.array(self.losses).mean()
        self.ys, self.ts = np.concatenate(self.ys), np.concatenate(self.ts)
        score = classification_metrics(self.ys, self.ts)
        summary = dict(name=self.name, loss=loss, **score)
        if len(score) > 0:
            if self.summary is None:
                self.summary = pd.DataFrame([summary], index=[epoch])
                self.summary.index.name = 'epoch'
            else:
                self.summary.loc[epoch] = summary
        if self.best_epoch == epoch:
            self.best_ys = self.ys
            self.best_ts = self.ts
        self.initialize()

    def get_dict(self):
        loss, fbeta, epoch = 0, 0, 0
        if self.summary is not None:
            row = self.summary.iloc[-1]
            epoch = row.name
            loss = row.loss
            fbeta = row.fbeta
        return {
            'epoch': epoch,
            'loss': loss,
            'fbeta': fbeta,
        }

    @property
    def fbeta(self):
        if self.summary is None:
            return 0
        else:
            return self.summary.fbeta[-1]

    @property
    def best_fbeta(self):
        return self.summary[self.main_metrics].max()

    @property
    def best_epoch(self):
        return self.summary[self.main_metrics].idxmax()

    @property
    def best_threshold(self):
        idx = self.summary[self.main_metrics].idxmax()
        return self.summary['threshold'][idx]
		
		
###############%%cython
get_ipython().run_cell_magic('cython', '', 'import numpy as np\ncimport numpy as np\nfrom multiprocessing import Pool\n\n\ncdef class ApplyNdArray:\n    cdef func\n    cdef dtype\n    cdef dims\n    cdef int processes\n\n    def __init__(self, func, processes=1, dtype=object, dims=None):\n        self.func = func\n        self.processes = processes\n        self.dtype = dtype\n        self.dims = dims\n\n    def __call__(self, arr):\n        if self.processes == 1:\n            return self.apply(arr)\n        else:\n            return self.apply_parallel(arr)\n\n    cpdef apply(self, arr):\n        cdef int i\n        cdef int n = len(arr)\n        if self.dims is not None:\n            shape = (n, *self.dims)\n        else:\n            shape = n\n        cdef res = np.empty(shape, dtype=self.dtype)\n        for i in range(n):\n            res[i] = self.func(arr[i])\n        return res\n\n    cpdef apply_parallel(self, arr):\n        cdef list arrs = np.array_split(arr, self.processes)\n        with Pool(processes=self.processes) as pool:\n            outputs = pool.map(self.apply, arrs)\n        return np.concatenate(outputs, axis=0)')
		
def load_module(filename):
    assert isinstance(filename, Path)
    name = filename.stem
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[mod.__name__] = mod
    return mod


def rmtree_after_confirmation(path, force=False):
    if Path(path).exists():
        if not force and not prompter.yesno('Overwrite %s?' % path):
            sys.exit(0)
        else:
            shutil.rmtree(path)


def pad_sequence(xs, length, padding_value=0):
    assert isinstance(xs, list)
    n_padding = length - len(xs)
    return np.array(xs + [padding_value] * n_padding, 'i')[:length]


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Pipeline(object):

    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

class TextNormalizer(TextNormalizerPresets):
    pass


class TextTokenizer(TextTokenizerPresets):
    pass


class WordEmbeddingFeaturizer(WordEmbeddingFeaturizerPresets):
    pass


class WordExtraFeaturizer(WordExtraFeaturizerPresets):

    default_config = dict(
        word_extra_features=['idf'],
    )


class SentenceExtraFeaturizer(SentenceExtraFeaturizerPresets):

    default_config = dict(
        sentence_extra_features=['char', 'word'],
    )


class Preprocessor(PreprocessorPresets):

    embedding_sampling = 400

    def build_word_features(self, word_embedding_featurizer,
                            embedding_matrices, word_extra_features):
        embedding = np.stack(list(embedding_matrices.values()))

        # Concat embedding
        embedding = np.concatenate(embedding, axis=1)
        vocab = word_embedding_featurizer.vocab
        embedding[vocab.lfq & vocab.unk] = 0

        # Embedding random sampling
        n_embed = embedding.shape[1]
        n_select = self.embedding_sampling
        idx = np.random.permutation(n_embed)[:n_select]
        embedding = embedding[:, idx]

        word_features = np.concatenate(
            [embedding, word_extra_features], axis=1)
        return word_features


# =======  Training modules  =======

class Embedding(EmbeddingPresets):
    pass


class Encoder(EncoderPresets):
    pass


class Aggregator(AggregatorPresets):
    pass


class MLP(MLPPresets):
    pass


class Ensembler(EnsemblerPresets):
    pass

class ExperimentConfigBuilder(ExperimentConfigBuilderBase):

    default_config = dict(
        test=False,
        device=0,
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=4,
    )

    @property
    def modules(self):
        return [
            TextNormalizer,
            TextTokenizer,
            WordExtraFeaturizer,
            SentenceExtraFeaturizer,
#             Embedding,
#             Encoder,
#             Aggregator,
#             MLP,
        ]

config = ExperimentConfigBuilder().build(args=[])
print(config)
start = time.time()
set_seed(config.seed)

train_df, submit_df = load_qiqc(n_rows=config.n_rows)
datasets = build_datasets(train_df, submit_df, config.holdout, config.seed)
train_dataset, test_dataset, submit_dataset = datasets

print('Tokenize texts...')
preprocessor = Preprocessor()
normalizer = TextNormalizer(config)
tokenizer = TextTokenizer(config)
train_dataset.tokens, test_dataset.tokens, submit_dataset.tokens = \
    preprocessor.tokenize(datasets, normalizer, tokenizer)
	
print('Build vocabulary...')
vocab = preprocessor.build_vocab(datasets, config)

print('Build token ids...')
train_dataset.tids, test_dataset.tids, submit_dataset.tids = \
    preprocessor.build_tokenids(datasets, vocab, config)
	
print('Build sentence extra features...')
sentence_extra_featurizer = SentenceExtraFeaturizer(config)
train_dataset._X2, test_dataset._X2, submit_dataset._X2 = \
    preprocessor.build_sentence_features(
        datasets, sentence_extra_featurizer)
[d.build(config.device) for d in datasets]

print('Build word extra features...')
word_extra_featurizer = WordExtraFeaturizer(config, vocab)
word_extra_features = word_extra_featurizer(vocab)

print(word_extra_features.shape)

word_dic = {'word_extra_features':word_extra_features, 'vocab': vocab.word_freq}
savemat('./IDF.mat', word_dic)