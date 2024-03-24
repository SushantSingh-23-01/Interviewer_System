import colorama
import ollama, re, time, json
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
import numpy as np
from colorama import  Fore, Style
# Important: Keep Context Size of Model in Mind.

colorama.init(autoreset=True)

job_name = 'Natural'

prompt1 = f'''
Imagine you are an interviewer hiring for a {job_name}. Ask one question at a time. Keep the interview professional. Ask questions relevant to discussion. Keep the question as short as possible.
Following are the chat logs between you and the candidate.
'''
prompt2 = f'''
Come up with a follow up question for the context: 
'''

prompt3 = f'''
Do not act as an AI or an assistant. You are a professional in {job_name}. You are being interviewed for {job_name} postion. Give answers to the questions asked. Keep the answer as short as possible. Keep the answers relevant to the discussion.
Take help of following text to answer the questions. 
'''


def read_txt(path):
    return open(path,'r+').read()
    
def read_pdf(path):
    text = ""
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_json(path):
    f = open(path)
    data = json.load(f)
    temp = []
    for i in data:
        for _,v in i.items():
            temp.append(v)
    f.close()
    return temp

def text_splitter(text,chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def bm25_search(chunks:list,query:str):
    tokenized_corpus = [i.split(' ') for i in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_results = bm25.get_top_n(query.split(' '), chunks, n = 5)
    user_index = chunks.index(query_results[0])
    doc_scores = bm25.get_scores(query.split(' '))
    
    # See below for reason of normalising
    i = np.argsort(doc_scores)  # indicies of sorted array in ascending order
    sorted_arr = doc_scores[i]  # Sorted array
    sorted_arr = sorted_arr[-10:]
    sorted_arr /= np.linalg.norm(sorted_arr)
    user_score = max(sorted_arr)
    
    result_text = ''
    for i in range(user_index-1,user_index+1):
        result_text += chunks[i]
    
    return user_score, result_text

def llm_generate(prompt:str,user_input:str)->str:
    response = ollama.chat(model='gemma-2b-it', messages=[
    {
        'role': 'user',
        'content': f'{prompt}\n{user_input}',
    },
    ])
    result = str(response['message']['content'])
    result = result.replace('**', "") 
    return result

def preprocess(text):
    return re.sub(r"[^a-zA-Z?\"\']+",' ',text) 

def main():
    # Not recommended to load text on memory.
    data = read_pdf(path='')
    chunks = text_splitter(data)
    chat_log = ''
    user_input = 'Hello Nice to meet you.'
    log = []
    print(f'You are going to be interviewed by an AI bot. Please introduce yourself to start the interview.')
    for i in range(0,5):
        #user_input = input(f'{Fore.LIGHTRED_EX}\nYou:')
        if user_input.lower() == 'quit':
            break
        if len(chat_log) >= 2048:
            chat_log = chat_log[-2048:]    
        else:
            pass
        start = time.time()        
        question = llm_generate(prompt=prompt1,user_input=user_input+chat_log)
        print(f'{Fore.CYAN}\nInterviewer:{question}')
    
        _, document = bm25_search(chunks=chunks,query=question)
        answer = llm_generate(prompt=prompt3,user_input=question+'\nRelevant context'+document)  
        print(f'\nUser: {answer}')
        end = time.time()
        print(f'{end-start:0.6f} seconds') 
        
        chat_log += question + '\n' + answer + "\n"
    
        log.append({'que':question, 'ans': answer})
        
    with open('log.json','w') as file:
        json.dump(log,file,indent=4)
  
    
if __name__ == '__main__':
    main()
