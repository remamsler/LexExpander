#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import:
import LexExpander


# In[2]:


#create an LexExpander Instance ... 
# we also need an embedding file
# and optionally a steering|starting point: these are some candidates 
# that we initially give as a steering (where to search)

EMBEDDINGFILE = "smd_50k_kv"
LEXICON = "Tiere_minimal.txt"
STARTINGPOINT = ["Elefant", "Nashorn"]

my_expander = LexExpander.LexExpander(  given_embedding_file= EMBEDDINGFILE , 
                                        lexicon_file = LEXICON,
                                        candidates_list = STARTINGPOINT,
                                        # target_folder = ".",
                                        )


# In[3]:


#prepare: this means: load lexicons, embeddings, set stuff ...
my_expander.prepare(verbose=True)


# In[4]:


#and launch:
my_expander.run()


# In[6]:


my_expander.write_out(target_filename="tiere_not_so_minimal.txt")


# In[ ]:




