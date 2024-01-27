#!/usr/bin/env python
# coding: utf-8

# # Building Question Answering Application using OpenAI, Pinecone, and LangChain

# LLM models are great at answering questions but only on topics they have been trained on. However, it wouldn't understand our question if we want it to answer questions about topics it hasn't been trained on, such as recent events after September 2021. In this notebook we are going to build an application that will allow us to send questions to our document and private data and get answers

# # Library Installation

# In[2]:


get_ipython().system('pip install openai')
get_ipython().system('pip install langchain')
get_ipython().system('pip install pinecone-client')
get_ipython().system('pip install python-dotenv')
get_ipython().system('pip install tiktoken')
get_ipython().system('pip install pypdf -q')
get_ipython().system('pip install docx2txt -q')


# ### Loading Environment Variables

# In[3]:


import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

os.environ.get('OPENAI_API_KEY')

print("API Key Loaded:", os.environ.get('OPENAI_API_KEY') is not None)


# ### Function loading documents with different formats (pdf, docx)

# In[56]:


def extract_text_from_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None
    
    data = loader.load()
    return data


# ### Testing the function using PDF file

# In[57]:


data = extract_text_from_document('Files/GNNs.pdf')
print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[5].page_content)} characters in the page')
print(data[0].page_content)
print(data[5].metadata)


# ### Testing the function using word file

# In[58]:


data = extract_text_from_document('Files/Summary.docx')
print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[0].page_content)} characters in the page')
print(data[0].page_content)
print(data[0].metadata)


# ### Chunking Strategies and Splitting the Documents

# In[61]:


def split_text_into_chunks(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


# ### Calling the chunk function

# In[62]:


my_chunks = split_text_into_chunks(data)
print(len(chunks))
print(chunks[1].page_content)


# ### Calculating the cost

# In[63]:


def calculate_and_display_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD:{total_tokens / 1000 * 0.0004:.6f}')


# ### Load or create embedding index function

# In[64]:


def load_or_create_embeddings_index(index_name,chunks):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Done')
    else:
        print(f'Creating index {index_name} and embeddings ...', end = '')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name = index_name)
        print('Done')
        
    return vector_store


# ### Delete Pinecone Index Function

# In[65]:


def drop_pinecone_index(index_name='all'):
    import pinecone
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ...')
        for index in indexes:
            pinecone.delete_index(index)
        print('Done')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Done')


# ### Testing the Load or create embedding index function

# In[67]:


data = extract_text_from_document('Files/GNNs.pdf')
print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[5].page_content)} characters in the page')


# In[68]:


my_chunks = chunk_data(data)
print(len(chunks))


# In[69]:


calculate_and_display_embedding_cost(my_chunks)


# In[72]:


drop_pinecone_index()


# In[73]:


index_name='gnnsdocument'
vector_store = load_or_create_embeddings_index(index_name, my_chunks)


# ### Creating the Function for Questions and Answers

# In[74]:


def generate_answer_from_vector_store(vector_store, question):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(question)
    return answer


# ### Sending questions continuously

# In[75]:


import time
i = 1
print("Please type Quit or Exit to quit.")
while True:
    question = input(f'Question #{i}:')
    i = i + 1
    if question.lower() in ['quit', 'exit']:
        print('Exiting the application.... bye bye!')
        time.sleep(2)
        break
    
    answer = generate_answer_from_vector_store(vector_store, question)
    answer_content = answer['result']
    print(f'\nAnswer: {answer_content}')
    print(f'\n {"-" * 100}\n')


# ### Adding Memory (Chat History) to our application

# In[77]:


def conduct_conversation_with_context(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc.invoke({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    
    return result, chat_history


# ### Testing the chat history

# In[79]:


chat_history = []
question = "What is GNN?"
result, chat_history = conduct_conversation_with_context(vector_store, question, chat_history)
print(result['answer'])
print(chat_history)


# In[80]:


chat_history = []
question = "What is this document about?"
result, chat_history = conduct_conversation_with_context(vector_store, question, chat_history)
print(result['answer'])
print(chat_history)


# In[81]:


chat_history = []
question = "How it discusses the method?"
result, chat_history = conduct_conversation_with_context(vector_store, question, chat_history)
print(result['answer'])
print(chat_history)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




