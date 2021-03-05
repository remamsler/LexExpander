#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Author: Michi Amsler
## Date: 2018

from gensim.models import Word2Vec, KeyedVectors
from time import time
import fasttext
import numpy as np

from functools import lru_cache

# how to use this:
# import CachedEmbedding
# my_emb = CachedEmbedding.CachedEmbedding(
#                                           given_embedding_file="./smd_50k_kv",
#                                           cache_size_get_embedding=20000,
#                                           cache_size_most_similar=1000)
# my_emb.prepare()
# my_emb.get_most_similars_cached(positive = ("Hund", "Katze"))
# my_emb.get_embedding_cached("Hund")


class CachedEmbedding(object):

    def __init__(self,
                 given_embedding_file=None,
                 embedding_model=None,
                 embedding_style="w2v",
                 cache_size_get_embedding=10000,
                 cache_size_most_similar=10000,
                 language=None,
                 **kwargs,
                 ):

        self.given_embedding_file = given_embedding_file
        self.embedding_model = embedding_model
        self.embedding_style = embedding_style
        self.cache_size_get_embedding = cache_size_get_embedding
        self.cache_size_most_similar = cache_size_most_similar
        self.language = language

    def load_embeddings(self, given_model_name=None, mode="w2v", language=None):
        """wrapper for embedding loader
        """

        # check if we have an overwrite:
        if given_model_name is not None:
            model_file_to_read_from = given_model_name
        else:
            model_file_to_read_from = self.given_embedding_file

        if mode == "w2vkv":
            print("we have a keyed vector model")
            self.load_w2v_model(given_model_name=model_file_to_read_from, kv=True)

        elif mode == "w2v":
            self.load_w2v_model(given_model_name=model_file_to_read_from)

        elif mode == "plaintext":
            self.load_plain_text(model_file_to_read_from)

        elif mode == "fasttext":
            self.load_fasttext_model(model_file_to_read_from)

        elif mode == "pymagnitude":
            if language is None:
                self.load_pymagnitude_model(given_model_name=model_file_to_read_from)
            else:
                self.load_pymagnitude_model(
                    given_model_name=model_file_to_read_from, language=language)
        else:
            print("not yet implemented!")

        return

    def load_plain_text(self, given_model_name):
        print("loading plain text model model {} ...".format(given_model_name))
        self.embedding_model = KeyedVectors.load_word2vec_format(given_model_name,
                                                                 binary=False)
        print("Done")

    def load_fasttext_model(self, given_model_name):
        self.embedding_model = FastTextEmbedding(given_model_name)

    def load_w2v_model(self, given_model_name=None, kv=False):
        '''load models; simple wrapper'''

        t0 = time()

        if kv:
            print("loading with keyedvectors method")
            self.embedding_model = KeyedVectors.load(given_model_name, mmap="r")
            print("... done in %0.3fs." % (time() - t0))

            return

        print("loading w2v model {} ...".format(given_model_name))
        try:
            self.embedding_model = Word2Vec.load(given_model_name, mmap="r")
        except:
            print("trying loading with keyedvectors method")
            self.embedding_model = KeyedVectors.load(given_model_name, mmap="r")

        print("... done in %0.3fs." % (time() - t0))

        return

    def load_pymagnitude_model(self, given_model_name=None, language=None):
        '''load models; simple wrapper'''

        t0 = time()

        # ugly but from tut:
        import pymagnitude

        print("loading pymagnitude model {} ...".format(given_model_name))
        if language is None:
            self.embedding_model = pymagnitude.Magnitude(given_model_name)
        else:
            self.embedding_model = pymagnitude.Magnitude(given_model_name, language=language)
        print("... done in %0.3fs." % (time() - t0))

        print("initializing for most_similar-searches...")
        t0 = time()
        print(self.embedding_model.most_similar(positive=["test"]))
        print("... done in %0.3fs." % (time() - t0))

        return

    def prepare(self, with_given_model=False):

        if not with_given_model:
            # first load the embeddings; e.g., w2v or pymagnitude
            self.load_embeddings(mode=self.embedding_style)
        else:
            # suppose we just need the wrapper; no loading needed:
            pass

        if self.embedding_style in ["w2v", "w2vkv", "plaintext"]:
            # we create this cached functions here, since we need the maxsize
            @lru_cache(maxsize=self.cache_size_get_embedding)
            def get_embedding(token_given=None):
                return self.embedding_model.wv[token_given]

            # Comment: since we try to cache the function
            # we have to call it with a TUPLE as input for the positive argument
            @lru_cache(maxsize=self.cache_size_most_similar)
            def get_most_similars(positive=None, topn=50):
                return self.embedding_model.wv.most_similar(positive=positive, topn=topn)

        elif self.embedding_style == "pymagnitude":

            # we create this cached functions here, since we need the maxsize
            @lru_cache(maxsize=self.cache_size_get_embedding)
            def get_embedding(token_given=None):
                return self.embedding_model.query(token_given)

            # Comment: since we try to cache the function
            # we have to call it with a TUPLE as input for the positive argument
            @lru_cache(maxsize=self.cache_size_most_similar)
            def get_most_similars(positive=None, topn=50):
                return self.embedding_model.most_similar(positive=list(positive), topn=topn)

        elif self.embedding_style == "fasttext":
            @lru_cache(maxsize=self.cache_size_get_embedding)
            def get_embedding(token_given=None):
                return self.embedding_model.get_vector(token_given)

            @lru_cache(maxsize=self.cache_size_most_similar)
            def get_most_similars(positive=None, topn=50):
                return self.embedding_model.get_nearest_neighbors(seed_set=positive,
                                                                  k=topn,
                                                                  exclude_seedset=False)

        # make them available
        self.get_embedding_cached = get_embedding
        self.get_most_similars_cached = get_most_similars

        return


class FastTextEmbedding(object):
    "Code largely taken from https://github.com/facebookresearch/fastText/issues/384"

    def __init__(self, model_file):
        self.model = fasttext.load_model(model_file)
        self.words = self.model.get_words()
        self.word_vectors = np.array([self.model[w] for w in self.words])

    def get_nearest_neighbors(self, seed_set, k=10, exclude_seedset=True):
        query = np.sum([self.model[w] for w in seed_set], axis=0)
        norms = np.sqrt((query**2).sum() * (self.word_vectors**2).sum(axis=1))
        cossims = np.matmul(self.word_vectors, query) / norms
        n_to_sort = k
        if exclude_seedset:
            n_to_sort += len(seed_set)
        rank = range(len(cossims)-n_to_sort, len(cossims))
        result_idx = np.argpartition(cossims, rank)[-n_to_sort:][::-1]
        result = [(self.words[idx], cossims[idx])
                  for idx in result_idx
                  if self.words[idx] not in seed_set or not exclude_seedset]
        if exclude_seedset:
            result = result[:k]
        return result

    def get_vector(self, token):
        return self.model[token]
