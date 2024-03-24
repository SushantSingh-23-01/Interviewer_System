import requests, wikipedia, re, random
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from collections import Counter
import ollama
import time, json

# Smaller Models have hard time falling prompt. This may lead to regex not finding any matches throwing error.

# This topics should be availble on wikipedia.
topics = [
    'Linear regression','Logistic regression','Polynomial regression','neural network','Support vector machine','Ridge regression','Decision tree','supervised learning',
    'Bagging','Boosting','Random forest','k-NN','Naive Bayes','Perceptron','Clustering','k-means','DBSCAN','LDA','PCA','t-SNE','Markov'
]

path = ''## Local Doc path

def preprocess(text):  
    text =  re.compile(r'<[^>]+>').sub('', text)
    text = re.sub(r'[^.,?a-zA-Z0-9]+',' ',text) 
    
    return text

def get_content_from_url(topics):
    temp = topics[random.randint(0,len(topics))]
    url = f'https://en.wikipedia.org/wiki/{temp}'
    page = requests.get(url)
    soup = BeautifulSoup(page.text,'lxml')
    # Inspect to find text body in browser
    text = soup.find('div',class_ = 'mw-page-container').text.strip() 
    text = preprocess(text)
    return text

def get_text_from_wikipedia(topic):
    content = wikipedia.page(topic).content
    return content

def read_pdf(path):
    text = ""
    pdf_reader = PdfReader(path)
    index = random.randint(0,len(pdf_reader.pages))
    for page in pdf_reader.pages[index:index+2]:
        text += page.extract_text()
    return text


def text_data_retrieval(text):
    tokens = re.split(pattern= r'[\b\W\b]+',string=text)
    frequency = Counter(tokens)
    top_k = frequency.most_common(20)
    matched_topics = []
    for i in topics:
        match = re.search(i.lower(),text)
        if match:
            print(f'Match found for {match.group()}')
            matched_topics.append(i)
        else:
            print(f'No match found for {i}')
    return top_k, matched_topics

def text_splitter(text,chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def scrapping_agent(context):
    prompt = f'''
    Write a question answer pair on the context provided.The question and answer generated should be detailed and coherent. The question generated should not be ambigious.
    Context: {context}
    The output should be in following format:
    Question: Qustion generated for given context
    Answer: Answer to the question from context.
    '''
    return prompt

def llm_gen(prompt):
    response = ollama.chat(model='gemma:2b', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    result = str(response['message']['content'])
    return result

def parse_llm_output(text):
    text = preprocess(text)
    que_match = re.search(pattern = 'Question', string = text)
    ans_match = re.search(pattern='Answer',string=text)
    parsed_que = text[que_match.end()+1:ans_match.start()-1]
    parsed_ans = text[ans_match.end()+1:]
    qa_dict = {'Question':parsed_que,'Answer':parsed_ans}
    return qa_dict

def main():
    start = time.time()
    data = []
    for _ in range(0,5):
        
        text = get_content_from_url(topics)
        chunks = text_splitter(text)
        indice = random.randint(0,len(chunks)-1)
        prompt = scrapping_agent(context=chunks[indice-1:indice+1])
        reply = llm_gen(prompt)
        print('-'*100)
        print(f'\n{reply}')
        qa_dict = parse_llm_output(reply)
        data.append(qa_dict)
    
    end = time.time()
    print(f"The time of execution of above program is :{(end-start):.4f}s")
    with open('qa_llm.json','w') as file:
        json.dump(data,file,indent=4)

if __name__ == '__main__':
    main()
