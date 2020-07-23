#!/usr/bin/python3
# -*- coding: utf-8 -*-

## Author: Michi Amsler
## Date: 2018

import sys, os
from time import time
import random
from collections import Counter

from copy import deepcopy

from functools import lru_cache

#from gensim.models import Word2Vec, KeyedVectors

import CachedEmbedding


class LexExpander(object):
    '''
    this class is used to expand a given lexicon in an iterative way, using a 
    generic model of a language in the form of embeddings, computed by e.g. word2vec
    '''

    def __init__(self, 
                    given_embedding_file = None,
                    embedding_model = None,
                    embedding_style = "w2v",
                    lexicon_file = None,
                    not_to_include_lexicon_file = None,
                    iterations = 10,
                    #
                    keep_size = 1,
                    new_in_lex_topn_to_draw_from = 4,
                    sample_size = 1, 
                    new_topn_to_draw_from = 2,
                    #
                    result_size = 100,
                    #
                    rec_runs = 3,
                    #
                    weak_add_rate = 0.5,
                    evaluation_threshold = 0.1,
                    #given_tonality = "DUMMY",
                    lowercasing = False,         
                    #
                    candidates_list = None,
                    target_folder = ".",
                 ):

        self.given_embedding_file = given_embedding_file
        self.lexicon_file = lexicon_file
        self.not_to_include_lexicon_file = not_to_include_lexicon_file
        self.iterations = iterations
        self.keep_size = keep_size
        self.new_in_lex_topn_to_draw_from =  new_in_lex_topn_to_draw_from
        self.sample_size =  sample_size
        self.new_topn_to_draw_from =  new_topn_to_draw_from
        self.result_size =  result_size
        self.rec_runs = rec_runs
        self.weak_add_rate = weak_add_rate
        self.evaluation_threshold = evaluation_threshold
        #self.given_tonality = given_tonality
        self.lowercasing = lowercasing
        self.candidates_list = candidates_list
        
        #we can also randomly sample
        self.random_sample = False 

        self.target_folder = target_folder

        #list of globals that we need:
        self.loaded_lex_as_set = None

        self.not_to_include_lexicon_loaded_lex_as_set = set()

        self.given_lexicon_set_WEAK = set()
        self.added_to_lexicon = set()

        self.embedding_model = embedding_model
        self.embedding_style = embedding_style

        #this is an attempt for a simple weighting scheme:
        #the more similar the term is (considering the position) the more weight i gains
        # 30 because of the  topn=30
        # maps 0 --> 30 .... 28 --> 2,  29 --> 1
        self.weighting_mapper = {i:30-i for i in range(30)}

        #temp run updates:
        self.added_in_this_run_weak = set()
        self.added_in_this_run_strong = set()
        
        #these are for the results of the recurrent runs:
        self.added_in_recurrent_runs_weak = set()
        self.added_in_recurrent_runs_strong =  set()

        #undo functionality
        self.old_attributes = None
        
        #debug/in-depth observation
        self.stop_after_each_step = False
        
        #steering verbosity
        self.report_lexicon_content = False


    def load_embeddings(self, given_model_name = None, mode = "w2v", **kwargs):
        """wrapper for embedding loader
        """
        
        #check if we have an overwrite:
        if given_model_name is not None:
            model_file_to_read_from = given_model_name
        else:
            model_file_to_read_from = self.given_embedding_file

        #we get a cached embedding wrapper
        ce = CachedEmbedding.CachedEmbedding(given_embedding_file=model_file_to_read_from, 
                                            embedding_style=mode,
                                            **kwargs,
                                            #cache_size_get_embedding=20000,
                                            #cache_size_most_similar=20000
                                            )

        ce.prepare()

        #we assign it to the object embedding_model attribute:
        #Attention: to enable the cache-functionality, we call the wrapper not the model directly
        # so if something else than get_embedding_cached() or get_most_similars_cached() should be used
        # we must then used self.embedding_model.embedding_model.METHOD
        self.embedding_model = ce

        return



    #load the dicts
    def load_lexicon(self, given_lexicon_file = None, flush_already_added = True, nonfile_mode_iterable = None):
        '''loads the lexicon and makes a set out of it'''

        print("reading in the lexicon ...")
        
        #given iterable mode
        if nonfile_mode_iterable is not None:
            #we expect to have a iterable
            self.loaded_lex_as_set = {element for element in nonfile_mode_iterable}
            
        # end given iterable mode
        else:
            #check if we have an overwrite:
            if given_lexicon_file is not None:
                lexicon_file_to_read_from = given_lexicon_file
            else:
                lexicon_file_to_read_from = self.lexicon_file

            t0 = time()

            with open(lexicon_file_to_read_from, "r", encoding="utf-8") as lex_file:
                if self.lowercasing:
                    loaded_lex_as_set = { line.rstrip().lower() for line in lex_file if line.strip() and not line.startswith("#")}
                else:
                    loaded_lex_as_set = { line.rstrip() for line in lex_file if line.strip() and not line.startswith("#")}
            
            print("... done in %0.3fs." % (time() - t0))

            self.loaded_lex_as_set = loaded_lex_as_set

        #since we are loading a lexicon freshly, we assume that we don't proceed
        # can be overwritten for "mixing" resources
        if flush_already_added:
            self.added_to_lexicon = set()

        return
    
    def load_not_to_include_lexicon(self, given_lexicon_file = None, nonfile_mode_iterable = None):
        '''loads the lexicon and makes a set out of it'''

        print("reading in the not-to-include-lexicon ...")
        
        #given iterable mode
        if nonfile_mode_iterable is not None:
            #we expect to have a iterable
            self.not_to_include_lexicon_loaded_lex_as_set = {element for element in nonfile_mode_iterable}
            
        # end given iterable mode
        
        else:
            #check if we have an overwrite:
            if given_lexicon_file is not None:
                lexicon_file_to_read_from = given_lexicon_file
            else:
                lexicon_file_to_read_from = self.not_to_include_lexicon_file

            t0 = time()

            with open(lexicon_file_to_read_from, "r", encoding="utf-8") as lex_file:
                if self.lowercasing:
                    not_to_include_lexicon_loaded_lex_as_set = { line.rstrip().lower() for line in lex_file if line.strip() and not line.startswith("#")}
                else:
                    not_to_include_lexicon_loaded_lex_as_set = { line.rstrip() for line in lex_file if line.strip() and not line.startswith("#")}
            
            print("... done in %0.3fs." % (time() - t0))


            self.not_to_include_lexicon_loaded_lex_as_set = not_to_include_lexicon_loaded_lex_as_set

        return

    def add_lexicon(self, given_lexicon_file = None, nonfile_mode_iterable = None):

        #given iterable mode
        if nonfile_mode_iterable is not None:
            #we expect to have a iterable
            to_add_loaded_lex_as_set = {element for element in nonfile_mode_iterable}

        #normal mode: expect given lex; read in; then add 
        else:
            if given_lexicon_file is not None:
                lexicon_file_to_read_from = given_lexicon_file
            else:
                print("no lexicon given!")
                return
            
            t0 = time()

            with open(lexicon_file_to_read_from, "r", encoding="utf-8") as lex_file:
                if self.lowercasing:
                    to_add_loaded_lex_as_set = { line.rstrip().lower() for line in lex_file if line.strip() and not line.startswith("#")}
                else:
                    to_add_loaded_lex_as_set = { line.rstrip() for line in lex_file if line.strip() and not line.startswith("#")}
            
            print("... done in %0.3fs." % (time() - t0))

        #updating/adding:
        self.loaded_lex_as_set.update(to_add_loaded_lex_as_set)

        return


    def add_not_to_include_lex(self, given_lexicon_file = None, nonfile_mode_iterable = None):

        #given iterable mode
        if nonfile_mode_iterable is not None:
            #we expect to have a iterable
            not_to_include_lexicon_loaded_lex_as_set = {element for element in nonfile_mode_iterable}

        #normal mode: expect given lex; read in; then add 
        else:
            if given_lexicon_file is not None:
                lexicon_file_to_read_from = given_lexicon_file
            else:
                print("no lexicon given!")
                return
            
            t0 = time()

            with open(lexicon_file_to_read_from, "r", encoding="utf-8") as lex_file:
                if self.lowercasing:
                    not_to_include_lexicon_loaded_lex_as_set = { line.rstrip().lower() for line in lex_file if line.strip() and not line.startswith("#")}
                else:
                    not_to_include_lexicon_loaded_lex_as_set = { line.rstrip() for line in lex_file if line.strip() and not line.startswith("#")}
            
            print("... done in %0.3fs." % (time() - t0))

        #updating/adding:
        self.not_to_include_lexicon_loaded_lex_as_set.update(not_to_include_lexicon_loaded_lex_as_set)

        return


#######################


#helper functions

    def show_config(self):
        """show brief summary of config of the expander:
        """
        print (""" 
            {} iterations
            {} keep_size
            {} new_in_lex_topn_to_draw_from
            {} sample_size
            {} new_topn_to_draw_from
            {} result_size
            {} rec_runs
            {} weak_add_rate
            
            """.format(self.iterations,
            self.keep_size, 
            self.new_in_lex_topn_to_draw_from, 
            self.sample_size, 
            self.new_topn_to_draw_from , 
            self.result_size, 
            self.rec_runs, 
            self.weak_add_rate, 
            ))

        print (""" 
            self.lexicon_file:\t{}
            embedding_model:\t{}
            candidates_list:\t{}
            
            target_folder:\t{}
            """.format(
            self.lexicon_file,
            self.embedding_model, 
            self.candidates_list,
            
            self.target_folder,
            ))                    

        
        return


    def get_tendency_weighted_general(self, test_candidates):
        '''return for a given candidate(or list of candidates)
        the majority vote (float) of how many in the top 10 next similars are already in the lexicon

        the question is if we can guess reliably if the candidates are good candidates for the lexicon

        ATTENTION: input should be a list
        '''
        tendency = "unknown"

        # get similars
        # most_similar_to_candidate_list = self.embedding_model.wv.most_similar(positive=test_candidates, topn=30)
        most_similar_to_candidate_list = self.embedding_model.get_most_similars_cached(positive=tuple(test_candidates), topn=30)

        #this are lists used for the sorting process below

        in_lexicon_list = []

        not_in_lexicon_list = []

        unknown_list = []

        # index also starting at 0
        for index, word_sim_pair in enumerate(most_similar_to_candidate_list):

            # this would just be a cut off
            #if (len(in_lexicon_list) + len(not_in_lexicon_list)) > 9:
            #    break

            #print(u"checking now {}".format(el))
            word = word_sim_pair[0]
            # if it's there at all ...
            if (word in self.loaded_lex_as_set):
                #print(u"found {} in LEX".format(word))
                in_lexicon_list.append((word, self.weighting_mapper[index]))
            # the case when its only weak ... half the weight
            elif word in self.given_lexicon_set_WEAK:
                #print(u"found {} in WEAK".format(word))
                
                in_lexicon_list.append((word, self.weighting_mapper[index] * self.weak_add_rate))

            else:
                unknown_list.append((word, self.weighting_mapper[index]))

        # gives us a bare number
        #neg_likelihood = sum([weight for (word, weight) in neg_list])
        #pos_likelihood = sum([weight for (word, weight) in pos_list])

        

        lexikon_likelihood = sum([weight for (word, weight) in in_lexicon_list ])
        unknown_likelihood = sum([weight for (word, weight) in unknown_list])



        in_lexicon_word_list = [word for (word, weight) in in_lexicon_list]
        not_in_lexicon_word_list = [word for (word, weight) in unknown_list]

        # NEW with unknowns
        # could be division by zero --- actually not
        try:
            lexicon_tendency = len(in_lexicon_list) / (len(in_lexicon_list) + len(unknown_list))
            # which is the same as: len(in_lexicon_list) / lne(most_similar_to_candidate_list)
            
        # if we have nothing related found --> 0.0
        except:
            lexicon_tendency = 0.0
            #pos_tendency = 0.0

        # return (neg_tendency, pos_tendency)
        return (lexicon_tendency, in_lexicon_word_list, not_in_lexicon_word_list, lexikon_likelihood, unknown_likelihood)


    def write_out(self, target_filename=None, target_folder=None, only_extension=False, include_weaks = False, mode = "new"):
        """writes out the expanded lexicon to the target
        
        Arguments:
            target_filename {[type]} -- [description]
        """
        #override
        if target_folder is not None:
            self.target_folder = target_folder
        
        set_to_write_out = set()
        
        # now the main part
        if only_extension:
            set_to_write_out.update(self.added_to_lexicon)
        else:
            set_to_write_out.update(self.loaded_lex_as_set)

        #
        if mode == "append":
            #we only want to append new ones:
            #this is the same as when setting parameter only_extension to True
            set_to_write_out = set()
            set_to_write_out.update(self.added_to_lexicon)

        # if also weaks should be integrated
        if include_weaks:
            set_to_write_out.update(self.given_lexicon_set_WEAK)
        

        print("checking targetfolder")
        #check if targetfolder exists; if not create:
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
            print("{} created".format(self.target_folder))


        if mode == "new":
            # write out to file 
            with open(os.path.join(self.target_folder, target_filename), "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(sorted([w for w in set_to_write_out], key=lambda s: s.casefold() )))
                outfile.write("\n")
        elif mode == "append":
            # append to file 
            with open(os.path.join(self.target_folder, target_filename), "a", encoding="utf-8") as outfile:
                #outfile.write("\n")
                outfile.write("\n".join(sorted([w for w in set_to_write_out], key=lambda s: s.casefold() )))
                outfile.write("\n")
        
        return 


    def printer_function(self, to_print_lists):
        
        already_in_list, same_polarity_list, different_polarity_list, new_candidates = to_print_lists
        
        print("***"*20)
        print("already in: {}".format(already_in_list))
        print("***"*20)
        print("SAME tonality: {}".format(same_polarity_list))
        print("---"*20)
        print("DIFFERENT tonality: {}".format(different_polarity_list))
        print("***"*20)
        print("NEW: {}".format(new_candidates))


    def flush(self):
        '''this is useful if we want to get clear states'''
        self.added_to_lexicon = set()
        self.given_lexicon_set_WEAK = set()
        
        self.load_lexicon()
        
        if self.not_to_include_lexicon_file is not None:
            self.load_not_to_include_lexicon()

        return
    
    def retract(self, given_list_of_words, add_to_not_include=False):

        for word in given_list_of_words:
            #using discard does not yield an error if not found in set
            self.added_to_lexicon.discard(word)
            self.loaded_lex_as_set.discard(word)
            print("removed {}".format(word))
        
        if add_to_not_include:
            self.add_not_to_include_lex(nonfile_mode_iterable = given_list_of_words)
            print("added {} to not_to_include_list".format(given_list_of_words))
            

    def get_results_for_candidates_list(self, candidates):
        # we get a candiate seed (list) to start with ...
        # e.g. candidates = [u"herrschsüchtig",u"herzlos", u"niederträchtig"]

        # now getting the topn-most similar, with the canidates as base; this is a list of tuples: (word, sim-metric)
        #the topn = 50 is somewhat arbitrary ...
        try:
            #most_similars_list = self.embedding_model.wv.most_similar(positive=candidates, topn=50)
            most_similars_list = self.embedding_model.get_most_similars_cached(positive=tuple(candidates), topn=50)
        #except KeyError, e:
        #    print(e)
        except KeyError as myerr:
            print(myerr)
        
            # ugly but gets the unknown word
            unknown_word = str(myerr).split("'")[1]
            
            # retract it from candidates
            candidates.remove(unknown_word)
            print("removed unknnown word {} from candidate list".format(unknown_word))
            
            #then call the method with the remaining list again
            print("starting now search-cycle now only for {}".format(u",".join(candidates)))
            # do it again
            return self.get_results_for_candidates_list(candidates)

        # most_similars_dict =  { t_element[0]:t_element[1] for t_element in w2v_model.most_similar(positive=candidates, topn=50)}
        # maybe check them first if in vocab:
        # if X in w2v_model.vocab: add to candidates

        # print("now we have the candidates: let's check them...")

        # keep the results apart:
        # 1. candidates for new
        new_candidates = []
        # 2. those which are already in the lexicon
        already_in_list = []
        # 3. those who seem to candidates for "IN"
        same_kind_list = []
        # 4. those who seems to be candidates for "OUT"
        different_kind_list = []

        # we proceed from the most_similar downwards (in terms of similarity)
        # look-up: already there?
        for el in most_similars_list:
            #print(u"checking now {}".format(el))
            word = el[0]
            if word in self.not_to_include_lexicon_loaded_lex_as_set:
                #skip this
                print("{} in the NOT-TO-INCLUDE-LEXICON".format(word))
                continue
            elif word in self.loaded_lex_as_set:
                print("{} already in given LEXICON".format(word))
                already_in_list.append(word)
                #we must add it here so it ends of in the confirmed list
                same_kind_list.append(word)
                # polarity check:
                # if POLEX_set[word][0].endswith(given_tonality):
                #     continue
                #     #same_kind_list.append(word)
                # # the neutrals:
                # elif POLEX_set[word][0].endswith("NEU"):
                #     continue
                # else:
                #     continue
                #     #different_kind_list.append(word)
            else:
                #print(u"not found in given LEXICON: {}".format(word))
                new_candidates.append(word)

        #since it makes not much sense, we leave same_kind_list and differente_kind_list empty
        return (already_in_list, same_kind_list, different_kind_list, new_candidates)


    def do_search(self, cands):
        '''does the cycle: take seed; look-up; return_results; resample
        kind of a wrapper right now
        '''
        
        print("starting search-cycle for {}".format(u",".join(cands)))
        
        #get results
        # (already_in_list, same_kind_list, different_kind_list, new_candidates)
        result_lists_tuple = self.get_results_for_candidates_list(cands)
        
        #show
        #printer_function(result_lists_tuple)

        #the already-in the LEXICON
        already_in = result_lists_tuple[0]
        print("already in lexicon: {}".format(len(already_in)))
        
        #those which could be confirmed
        confirmed = result_lists_tuple[1]

        #NOT SO INTERESTING FOR THE MOMENT ...
        #the already-in with different-tonality: those should be checked - see below
        diff = result_lists_tuple[2]
        #these are not in the lexicon
        new = result_lists_tuple[3]


        #remember
        return (already_in, confirmed, new, diff)
        

    def do_process(self, run_number, evaluation_verbosity=2):
        
        #we re-set them empty here because they belong to single runs (not recurrent runs)
        self.added_in_this_run_weak = set()
        self.added_in_this_run_strong = set()

        print("we start with run number {}".format(run_number))
        
        #now inform about extension:
        print("in the lexicon we have:")
        if self.report_lexicon_content:
            print(sorted(self.loaded_lex_as_set))
        else:
            print("{} entries till now...".format(len(self.loaded_lex_as_set)))
        
        print("in the weak lexicon we have:")
        if self.report_lexicon_content:
            print(sorted(self.given_lexicon_set_WEAK))
        else:
            print("{} entries till now...".format(len(self.given_lexicon_set_WEAK)))
        
        
        if self.not_to_include_lexicon_loaded_lex_as_set is not None:
            print("in the not-to-inlcude-lexicon we have:")
            if self.report_lexicon_content:
                print(sorted(self.not_to_include_lexicon_loaded_lex_as_set))
            else:
                print("{} entries till now...".format(len(self.not_to_include_lexicon_loaded_lex_as_set)))
            
        
        #we collect here what to add
        to_add_to_lex_list = []
        used_candidates = []
        diff_list = []
        confirmed_lists = []
        to_retract_candidates = []


        #here we do the stuff iterations-times ...
        # i.e. we collect a lot candidates, several times, then add them up
        
        new_try_cands = []

        for i in range(self.iterations):
            #do it:
            #for the first round: use the given seed:
            
            if i == 0:
                #seed ...
                #cands = [u"herrschsüchtig",u"herzlos", u"niederträchtig"]
                cands = self.candidates_list
                #print("cands", cands)
            else:
                #when we do the iterations we take new candidates - based on the selection
                cands = new_try_cands

            #get back results:
            #ONLY new is interesting
            # conf are already in the lexicon
            # diff would be the "suspicious" ones
            
            # we get: (already_in, confirmed, new, diff)
            in_lex, conf, new, diff = self.do_search(cands)

            #ATTENTION!!!!
            #we set here conf to in_lex
            conf = in_lex

            #global appending:
            print("appending {} new candidates".format(str(len(new))))
            print(new)
            print("***"*20)
            to_add_to_lex_list.append(new)
            confirmed_lists.append(conf)
            used_candidates.append(cands)
            #only extend
            diff_list.extend(diff)
            #take sample, let's say 5:
            #new_try_cands = random.sample(conf, sample_size)
            

            #now new seed:
            # we take KEEP_SIZE from IN_LEX(cut_off_by_NEW_IN_LEX_TOPN_TO_DRAW_FROM) and add
            # SAMPLE_SIZE from NEW(cut_off_by_NEW_TOPN_TO_DRAW_FROM)
        

            #new_try_cands = random.sample(conf[:new_conf_topn_to_draw_from], keep_size) + random.sample(new[:new_topn_to_draw_from], sample_size)
            
            #change here for lexical extension:
            #####################################
            ## sampling for re-run:
            #####################################
            #for the in_lex part (the similar canidates which were already in the lexicon, ranked by similarity):
            # - we set a stopper (out-of-the-first-n): the new_in_lex_topn_to_draw_from
            # - we take from there N candidates:  keep_size
            # for the new part (the similar candidates which were NOT in the lexicon, ranked by similarity):
            # since they are ordered in similarity rank: we draw from the top again here (since those are the best candidates - similaritywise)
            # - we set a stopper (out-of-the-first-n): the new_topn_to_draw_from
            # - we take from there N candidates:  sample_size
            
            #sometimes there is an error with the random sample
            # this happens, when the list indexing is not working (cli params define where the boundaries should be ...)
            # or if we want to draw to many candidates
            # instead of check beforehand, we ask for permission and then do a recalibration of the sample drawing
            
            try:
                sample_from_inlex = random.sample(in_lex[:self.new_in_lex_topn_to_draw_from], self.keep_size) 
            except:
                print("Problem with drawing sample from known ones")
                print("keep_size:", self.keep_size)
                print("new_in_lex_topn_to_draw_from:", self.new_in_lex_topn_to_draw_from)
                print("len of in_lex: {}".format(len(in_lex)))
                #here we reset the sampleN to be drawn to min(keep_size, len(in_lex)) 
                # --> if sampleN (keep_size) is bigger than list, we take list-length number of examples
                if len(in_lex) != 0:
                    sample_from_inlex = random.sample(in_lex,min(self.keep_size, len(in_lex)))
                #if len of in_lex is 0, then we have nothing from here
                else:
                    #we get just one from the lexicon - which is a set - unordere, different at each call
                    sample_from_inlex = random.sample(self.loaded_lex_as_set,1)
                    print("drawn randomly from the lexicon: {}".format(sample_from_inlex[0]))
            try:
                sample_from_new_ones = random.sample(new[:self.new_topn_to_draw_from], self.sample_size)
            except:
                print("Problem with drawing sample from new ones")
                print("sample_size:", self.sample_size)
                print("new_topn_to_draw_from:", self.new_topn_to_draw_from)
                print("len of new: {}".format(len(new)))
                #here we reset the sampleN to be drawn to min(sample_size, len(new))
                # --> if sampleN (sample_size) is bigger than list, we take list-length number of examples
                sample_from_new_ones = random.sample(new[:len(new)], min(self.sample_size, len(new)))
            
            #merge:
            new_try_cands = sample_from_inlex + sample_from_new_ones
            
            ### this gives us a mixture (which we can guide via the commandline params) 
            # of such similar terms which are in the lexicon
            # and such ones that are drawn from the best new candidates 
                
                
            
        
            
            #needed?
            #this shouldn't hurt ..
            #if keep_size != 0:

                ##new_try_cands = random.sample(conf,min(keep_size, len(conf))) + random.sample(new[:len(new)], min(sample_size, len(new)))
                
                ##here we reset the sampleN to be drawn to min(keep_size, len(in_lex)) 
                ## --> if sampleN (keep_size) is bigger than list, we take list-length
            #else:
                #new_try_cands = in_lex[:1]
            ##new_try_cands = new[:5] 
            
            
            #stop, wait till ENTER is pressed
            if self.stop_after_each_step:
                input("Press Enter to continue...")
            
            #and go again ...
            continue

        print()
        print("&&&"*20)
        print()
        
        #we show only the ones from this turn 
        #print("and now se clap", run_number, iterations, run_number*iterations, len(to_add_to_lex_list[(run_number-1)*iterations:]), len(to_add_to_lex_list))
        #for index, round_list in enumerate(to_add_to_lex_list[(run_number-1)*iterations:]):
            #print(u"round {}: candidates were: {}\nnew: {}\nconfirmed this time: {}\n-----".format((run_number-1)*iterations+index+1, used_candidates[(run_number-1)*iterations+index] ,round_list, confirmed_lists[(run_number-1)*iterations+index]))

        #this is reporting about what has been collected as new candidates
        for index, round_list in enumerate(to_add_to_lex_list):
            print("run {}- iteration {}: candidates (seed words) were: {}\nnew: {}\nalready in list this time: {}\n-----".format(run_number ,index+1, used_candidates[index] ,round_list, confirmed_lists[index]))

        print()
        print("***"*20)
        print()


        #flattening and counting:
        flat_list = [item for sublist in to_add_to_lex_list for item in sublist]
        counted = Counter(flat_list)

        ################################################################################################################
        #print(u"we have {} candidates ...\ngiving here top {}:".format(len(counted), result_size))
        print("we have {} candidates ...\ngiving here top {}:".format(len(counted), len(counted)))

        #for i in counted.most_common(result_size):
        #print(u"\t".join([ u"{}:::{}".format(w,c) for (w,c) in counted.most_common(result_size)]))
        print("\t".join([ "{}:::{}".format(w,c) for (w,c) in counted.most_common(len(counted))]))
        
        #stop, wait till ENTER is pressed
        if self.stop_after_each_step:
            input("Press Enter to continue...")

        # now we check the tendency: are they to add or not:
        print()
        print("---tendency-check for new candidates:")
        print()
        t0 = time()
        for (w,c) in counted.most_common(self.result_size):
            #must be fed in as a lists
            #(neg_tend, pos_tend, n_list, p_list, n_lh, p_lh, u_lh) = get_tendency_weighted([w])

            #lexicon_tendency, in_lexicon_word_list, not_in_lexicon_word_list, lexikon_likelihood, unknown_likelihood

            (lex_in_tend, in_lex_w_list, not_in_lex_w_list, lex_lh, unknown_lh ) = self.get_tendency_weighted_general([w])
            #default
            verbal_tendency = "not decidable"
            
            #these are the percentages of what was achievable in general by this summed up counts
            lex_lh_perc = lex_lh/(lex_lh+unknown_lh)
            unknown_lh_perc = unknown_lh/(lex_lh+unknown_lh)
            
            #default is 2
            if evaluation_verbosity > 1:
            
                print("\nfor {}:".format(w))
                
                print("lex_in_simple (% of words in lex) = {}".format(lex_in_tend))
                
                print("lex_lh = {}".format(lex_lh))
                print("unknown_lh = {}".format(unknown_lh))
                
                print("lex_lh_perc = {}".format(lex_lh_perc))
                print("unknown_lh_perc = {}".format(unknown_lh_perc))
            
            
            #some factor ... if there is 10% or more of the "known-likelihood"
            #if lex_lh_perc > unknown_lh_perc/10:
            if lex_lh_perc > unknown_lh_perc * self.evaluation_threshold:
                
                #if it seems to be more known then not known
                if lex_lh_perc > unknown_lh_perc:
                    verbal_tendency = "very strong TO_ADD_IN"
                    if evaluation_verbosity > 0:
                        print("##### adding {} to lex! #####".format(w))
                    #for the coming runs, we will just assume, that this is already in the lexicon
                    self.loaded_lex_as_set.add(w)
                    # and here we keep track of what we have added
                    self.added_to_lexicon.add(w)
                    #for this run only:
                    self.added_in_this_run_strong.add(w)
                    #retract from weaks if there:
                    self.added_in_this_run_weak.discard(w)

                    #...and retract it from the WEAK dictionary (if in):
                    #if w in POLEX_set_WEAK_NEGs: --> not neede when using pop with default None
                    self.given_lexicon_set_WEAK.discard(w)
                    
                else:
                    verbal_tendency = "rather weak NEW"
                    #######################
                    ####daring ....
                                    
                    #if lex_lh_perc > unknown_lh_perc - 0.30:
                    
                    #NOTE 2019: this version (in contrast to the one (minus 30 percent) above) is just the same as the
                    # ratio between the percentages in the outer "if"; so this will always yield True
                    if lex_lh / unknown_lh > self.evaluation_threshold:
                        verbal_tendency = "weak NEW"
                        if evaluation_verbosity > 0:
                            print("### adding the WEAK {} to TEMPlex! ###".format(w))
                        
                        #for this run only
                        self.added_in_this_run_weak.add(w)
                        #for the global weak lexicon:
                        self.given_lexicon_set_WEAK.add(w)

            #update them to keep an eye on the aggregated results
            self.added_in_recurrent_runs_strong.update(self.added_in_this_run_strong)
            self.added_in_recurrent_runs_weak.update(self.added_in_this_run_weak)

            ##### end lexicon update #############################################################
            #############################################################################################
            




            # prints
            if evaluation_verbosity > 0:
                print("\t".join(["{}:::{}".format(w,str(c)),      # the word:::count
                                "{0:.2f}".format(lex_in_tend), # the simple tendency
                                "{0:.2f}".format(lex_lh_perc),  # the lex_lh_percentage with 2 digits after comma
                                str(lex_lh),                    # the lex_lh
                                "{0:.2f}".format(unknown_lh_perc), # the unknown_lh_percentage with 2 digits after comma
                                str(unknown_lh),                # the unknown_lh

                                #"{0:.2f}".format(unknown_lh),
                                #str(u_lh),
                                #for illustrative purposes:
                                ",".join(in_lex_w_list[:5]),    # 5 words from the related ones that were in the lex
                                "--vs.--",                     #
                                ",".join(not_in_lex_w_list[:5]),# 5 words from the related ones that were NOT in the lex
                                "--> {}".format(verbal_tendency)])) # and the verdict

        print("... done in %0.3fs." % (time() - t0))
    
        return

    def print_output(self):

        print("\nNew entries (respective to the given lexicon):")
        counter = 0

        # local scope; lists
        strong_new = []
        strong_new_unigram = []
        strong_new_bigram = []

        #given_lexicon_set_WEAK
        #given_lexicon_set

        for new_word in self.added_to_lexicon:
            #print(new_word)
            strong_new.append(new_word)
            if not u"_" in new_word:
                strong_new_unigram.append(new_word)
            else:
                strong_new_bigram.append(new_word)

        print("overall {} new STRONG entries.\n{} words and {} bigrams.\n".format(len(self.added_to_lexicon), len(strong_new_unigram), len(strong_new_bigram)))
        print("***"*20)
        #
        print("***"*10, "Lexicon status: NEW_STRONGs", "***"*10)
        #for el in strong_new:
        #    print(u'{}\t\tNEW_STRONG'.format(el))
        #
        for el in sorted(strong_new_unigram):
            print("{}\t\tNEW_STRONG".format(el))
        print("---"*20)
        for el in sorted(strong_new_bigram):
            print("{}\t\tNEW_STRONG".format(el))


        print("***"*20)
        print("overall {} new weak entries".format(len(self.given_lexicon_set_WEAK)))

        print("***"*10, u"NEW_WEAKs", "***"*10)

        for weak_entry in self.given_lexicon_set_WEAK:
            print("{}\t\tNEW_WEAK".format(weak_entry))


        #TODO change this: use split_ext from os.path ....

        ###added to persist the lexicon of new STRONGS:
        #the hack cuts off "_seed_list.txt"
        # with open("{}_extension.txt".format(self.lexicon_file[:-14]), "w") as extension_lex:
        #     for el in sorted(strong_new_unigram):
        #         #print("{}\t\tNEW_STRONG".format(el))
        #         extension_lex.write(el+"\n")
        #     #print("---"*20)
        #     extension_lex.write("###\n")
            
        #     for el in sorted(strong_new_bigram):
        #         #print("{}\t\tNEW_STRONG".format(el))
        #         extension_lex.write(el+"\n")

        # #added for unified list:

        # with open("{}_unified.txt".format(self.lexicon_file[:-14]), "w") as unified_lex:
            
        #     #the seed ...
        #     for line in open(self.lexicon_file, "r"):
        #         unified_lex.write(line)
            
        #     #plus NEW STRONG unigrams and bigrams
        #     for el in sorted(strong_new_unigram):
        #         #print("{}\t\tNEW_STRONG".format(el))
        #         unified_lex.write(el+"\n")
        #     #print("---"*20)
        #     unified_lex.write("###\n")
            
        #     for el in sorted(strong_new_bigram):
        #         #print("{}\t\tNEW_STRONG".format(el))
        #         unified_lex.write(el+"\n")

        return
        
        
    def save_state(self):
        '''this saves the important lists/sets/dicts to restore if needed'''
        self.old_attributes = { 
                                #since that is a deepcopy, we get recursive saves here...
                                'old_attributes' : deepcopy(self.old_attributes),
                                'added_to_lexicon' : deepcopy(self.added_to_lexicon),
                                'candidates_list' : deepcopy(self.candidates_list),
                                'given_lexicon_set_WEAK' : deepcopy(self.given_lexicon_set_WEAK),
                                'loaded_lex_as_set' : deepcopy(self.loaded_lex_as_set),
                                'added_in_this_run_weak' : deepcopy(self.added_in_this_run_weak),
                                'added_in_this_run_strong' : deepcopy(self.added_in_this_run_strong),
                                'added_in_recurrent_runs_weak' : deepcopy(self.added_in_recurrent_runs_weak),
                                'added_in_recurrent_runs_strong' : deepcopy(self.added_in_recurrent_runs_strong),
                            }
        return
    
    def undo(self):
        #hmmm do we need deepcopy here or are pointers to actual refercence good enough?
        self.added_to_lexicon = self.old_attributes['added_to_lexicon']
        self.candidates_list = self.old_attributes['candidates_list']
        self.given_lexicon_set_WEAK = self.old_attributes['given_lexicon_set_WEAK']
        self.loaded_lex_as_set = self.old_attributes['loaded_lex_as_set']
        self.added_in_this_run_weak = self.old_attributes['added_in_this_run_weak']
        self.added_in_this_run_strong = self.old_attributes['added_in_this_run_strong']
        self.added_in_recurrent_runs_weak = self.old_attributes['added_in_recurrent_runs_weak']
        self.added_in_recurrent_runs_strong = self.old_attributes['added_in_recurrent_runs_strong']

        #then back-updateing old_attributes
        self.old_attributes = self.old_attributes['old_attributes']

        return

########## main prog

    def prepare(self, verbose = True):
        """this is a wrapper that does several initialising steps: 

        - reading in the given lexicon (seed)
        - loading the embedding model
        
        """

        self.load_lexicon()
        
        if self.not_to_include_lexicon_file is not None:
            self.load_not_to_include_lexicon()


        #check if we have a given embedding model: then use it
        # if not: check if we were provided with a path to a model
        # if so: try to load this one:
        #load if nothing is provided
        print("setting up embeddings ...")
        if self.embedding_model is None:
            try:
                print("loading embeddings")

                #DEV:
                cache_params = { "cache_size_most_similar" : 30000,
                                "cache_size_get_embedding" : 15000,}

                self.load_embeddings(mode = self.embedding_style, **cache_params)
            except:
                if self.given_embedding_file is None:
                    print("please add an embedding model!")
        else:
            print("using passed model {}".format(self.embedding_model))
        
        if verbose:
            self.show_config()

        return

    @staticmethod
    def save_sampling(iterable, no_of_items): 
        try: 
            return random.sample(iterable, no_of_items) 
        except ValueError as error: 
            print(error) 
            print("will return a sample with size of population") 
            #then to the rescue: 
            return random.sample(iterable, len(iterable)) 


    def get_steering(self, mode = "random", sample_size = 2, given_sampling_population = None):
        '''this function provides the steering, i.e. the 
        terms that define the starting point for a run
        modes:
            - random: select randomly x elements out of the lexicon
            - cluster_random: first cluster lexicon; then apply recombination
            of nearest-to-centorid-terms
            
        returns: a set of terms (or a point in vector space)
        '''
        
        #we can also inject a specific sampling population:
        if given_sampling_population is not None:
            sampling_population = given_sampling_population
        # default is the lexicon:
        else:
            sampling_population = self.loaded_lex_as_set,
        
        
        if mode == "random":
            steering_terms = self.save_sampling(sampling_population, sample_size)
        elif mode == "cluster_random":
            pass

        return steering_terms


    def run(self, flush_weaks = True, 
                    steering_to_weak_lex=False, 
                    keep_for_undo = True, 
                    report = True,
                    sample = False,
                    sample_size = 2,
                    given_sampling_population = None, 
                    given_steering = None, 
                    stop_after_each_step=False, 
                    evaluation_verbosity=2):
        """this is wrapper which performs the main program
        """

        if keep_for_undo:
            self.save_state()

        #empty these temporary lists
        #self.added_in_this_run_weak = set()
        #self.added_in_this_run_strong = set()

        self.added_in_recurrent_runs_strong = set()
        self.added_in_recurrent_runs_weak = set()


        if flush_weaks:
            self.given_lexicon_set_WEAK = set()
        
        
        #this is the debug/step-by-step mode
        if stop_after_each_step:
            self.stop_after_each_step = True
        else:
            self.stop_after_each_step = False
        
                
           
        if sample == "random":
            #this gets randomly sample_size elements from lex
            #we just pass what we have as sampling_population; if None, this is handled in the function
            self.candidates_list = self.get_steering(mode = "random", sample_size = sample_size, given_sampling_population = given_sampling_population)
            
        elif sample == "cluster_random":
            print ("not yet implemented")
            
        elif sample == "cluster_centroid":
            print ("not yet implemented")
        
        elif sample == "given":
            if given_steering is not None:
                self.candidates_list = given_steering
            else:
                print("no steering provided; will use fallback!")
            
        ###just check if we have seed candidates; if not then we apply 
        # simple random sampling; 2 elements; this is a fallback
               
        if not self.candidates_list:
            self.candidates_list = random.sample(self.loaded_lex_as_set, 2)
            print("no seed candidates given; drawn 2 from lexicon:", self.candidates_list)
        
             
        
        #this is for blending of concepts; then we would like to use the given steering as 
        # weak evidence
        if steering_to_weak_lex:
            self.given_lexicon_set_WEAK.update(self.candidates_list)


        ######### main ###########

        print("iterations", self.iterations)
        print("rec_runs", self.rec_runs)

        for i in range(self.rec_runs):
            self.do_process(i+1, evaluation_verbosity = evaluation_verbosity)
            
            if evaluation_verbosity > 1:
                #after each run ...
                #thats not true ... or it mmust be emptied ...TODODOOOOOO
                print("---"*20)
                print("ADDED to LEX in this run:\n")
                print(self.added_in_this_run_strong)
                print("..."*20)
                print("ADDED to WEAK-LEX in this run:\n")
                print(self.added_in_this_run_weak)
                print("---"*20)
                
                #stop, wait till ENTER is pressed
                if self.stop_after_each_step:
                    input("Press Enter to continue...")


        #this is the result after the recurrent runs
        print("---"*20)
        print("ADDED to LEX in the {} recurrent run(s):\n".format(self.rec_runs))
        print(self.added_in_recurrent_runs_strong)
        print("..."*20)
        print("ADDED to WEAK-LEX in the {} recurrent run(s):\n".format(self.rec_runs))
        print(sorted(self.added_in_recurrent_runs_weak))
        print("---"*20)
        
        ##############################
        #### results and output
        
        if report:
            self.print_output()
        
        pass


#TODO:

# maybe empty the weak_Lexicon if new run is triggered
# make this available via parameter
# so this should be possible to steer when re-run...
