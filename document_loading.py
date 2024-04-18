#!/usr/bin/env python
# coding: utf-8

# ## Document Loading

# ### Retrieval augmented generation
# 
# In Retrieval augmented generation(RAG), an LLM retrieves contextual documetns from an external dataset as part of its execution.
# 
# This is useful if we want to ask questions about specific documents (e.g our PDFS, a set of videos, etc)

# In[1]:


import os
import openai
import sys
sys.path.append('../..')

# Load api key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)


# ## PDFs

# PDF from transcript: words and sentences are sometimes split unexpectedly.

# In[2]:


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()


# Each page is a document that contains text page_content and metadata.

# In[3]:


len(pages)


# In[4]:


page = pages[0]


# In[5]:


print(page.page_content[0:500])


# In[6]:


page.metadata


# ## YouTube

# In[7]:


from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


# In[8]:


# If not installed you can install from here easiest. 
get_ipython().system('pip install yt_dlp -U')
get_ipython().system('pip install pydub')


# Loading the below can take several minutes to complete.

# In[9]:


#!pip install yt-dlp -U
get_ipython().system('pip install ffprobe')
get_ipython().system('pip install ffmpeg')


# In[10]:


url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir = "docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),OpenAIWhisperParser())
docs = loader.load()


# In[ ]:


docs[0].page_content[0:500]


# ## URLs

# In[11]:


from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/getting-started.md")


# In[12]:


docs = loader.load()


# In[13]:


print(docs[0].page_content[:500])

