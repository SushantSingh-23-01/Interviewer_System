import ollama, re, json, time
from sentence_transformers import SentenceTransformer
from colorama import Fore, init # Colourful ouptut text
import matplotlib.pyplot as plt
from torch.nn import CosineSimilarity
import torch
from duckduckgo_search import DDGS

#Use gemma2b or other model for testing. 
#Use mistral or any better model for infernece

plt.style.use('ggplot')
init(autoreset=True)

model_id = "sentence-transformers/all-MiniLM-L6-v2" 
embed_model = SentenceTransformer(model_id)

job_name = 'Data Science'

def preprocess(text):
    return re.sub(r"[^a-zA-Z0-9?:.\/]+",' ',text) 

def evaluator_agent(question, answer):
    prompt = f'''
    Imagine you are a test evaluator. You specialise in {job_name}.Check weather answers match the question. Be critical. Judge with high precision. 
    Question : {question}
    Answer : {answer}
    Ouput the final analysis in following format.
    Score: Give a hyptothetical score between 0 and 10 depicting the correctness of the answer.
    Insights: Details of how you decided wether the answer is correct or incorrect
    '''
    prompt2 = f'''Does the answer explain everything asked in the question. Check if anything is unanswered. Be accurate
    Question: {question}
    Answer: {answer}
    Ouput the final analysis in following format.
    Score: A number between 0 and 10. 0 depicts the answer fails to explain anything in the question. 10 depicts the answer completely expains everything in the question.
    Insights: The chain of thoughts of how the evaluation is done and justification of Score.
    '''
    return prompt

def agent_answer(question):
    answer = f'''
    Provide an answer for the question given. The answer should be concise. 
    question: {question} 
    You use a tool to answer the question if you don't have an answer.
    You have access to the following tools:
    Search: Useful for when you need to answer questions about current events. You should ask targeted questions.
    Thought: You should always think about what to do.
    Action: The action to take.
    Action Input: "the input to the action, to be sent to the tool.
    '''
    answer = re.sub(r'\n',' ',answer)
    return llm_generate(prompt=answer)


def agent_comparator(user_ans,gen_ans):
    comparator = f'''
    Compare wether the similarity of the two text provided.
    text A : {user_ans}
    text B : {gen_ans}
    Ouput the final analysis in following format.
    Evaluation: A short description of difference in two text.
    '''
    return llm_generate(prompt=preprocess(comparator))

def read_json(path):
    f = open(path)
    data = json.load(f)
    que, ans = [], []
    for i in data:
        que.append(i.get('interviewer'))
        ans.append(i.get('user'))
    # Avoid unecessary questions and answers
    que = que[1:]         
    ans = ans[2:]  
    f.close()
    return ans, que

def search_engine(query):
    results = DDGS().text(query,max_results=3)
    temp = ''
    for i in results:
        temp += i.get('body')
    return temp

def text_splitter(text,chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def llm_generate(prompt:str)->str:
    response = ollama.chat(model='gemma-2b-it', messages=[
    {
        'role': 'user',
        'content': f'{prompt}',
    },
    ])
    result = str(response['message']['content'])
    result = result.replace('**', "") 
    return result

def parse_llm_output(text):
    text = preprocess(text)
#    que_match = re.search(pattern = 'Evaluation', string = text)
    match_score = re.search(pattern='Score: ',string=text)
    match_ans = re.search(pattern='Insights: ',string=text)
    
#    parsed_que = text[que_match.end()+1:ans_match.start()-1]
    score = text[match_score.end():match_ans.start()-1]
    insight = text[match_ans.end():]
    qa_dict = {'Score':score,'Insights':insight}

    return qa_dict

def extract_action_and_input(text):
    action_pattern = r"Action: (.+?)\n"
    input_pattern = r"Action Input: \"(.+?)\""
    action = re.findall(action_pattern, text)
    action_input = re.findall(input_pattern, text)
    return action, action_input

def cos_similarity(user_ans, gen_ans):
    user_ans = embed_model.encode(user_ans)
    gen_ans = embed_model.encode(gen_ans)
    cos =  CosineSimilarity(dim=-1)
    return cos(torch.from_numpy(user_ans),torch.from_numpy(gen_ans)).max().item()

def plot_histogram(data):
    plt.title('Historgram of scores out of 10')
    plt.hist(data)
    plt.show()    

def main():
    start = time.time()
    data ,scores = [],[]
    user_ans,que = read_json('log.json')
    for i in range(0,len(user_ans)):
        print('-'*100)
        print(f'\n{Fore.LIGHTBLUE_EX}Question: {que[i]}\n\n{Fore.LIGHTMAGENTA_EX}Answer: {user_ans[i]}')
        #prompt = evaluator_agent(question=que[i],answer=ans[i])
        
        gen_ans = agent_answer(question=que[i])
        print(f'\n{Fore.LIGHTGREEN_EX}Expected answer: {gen_ans}')
        
        action, action_input = extract_action_and_input(gen_ans)    
        
        if action == ['Search']:
            gen_ans = search_engine(action_input[0])
            print(gen_ans) 
        evaluation = agent_comparator(user_ans=preprocess(user_ans[i]),gen_ans=preprocess(gen_ans))
        print(f'\n{Fore.LIGHTRED_EX}Evaluation: {evaluation}')
        
        cos = cos_similarity(user_ans=preprocess(user_ans[i]),gen_ans=preprocess(gen_ans))
        print(f'\n{Fore.CYAN}Cosine Similarity: {cos}')
        data.append({'que':preprocess(que[i]),
                     'user_ans':preprocess(user_ans[i]),
                     'gen_ans':preprocess(gen_ans),
                     'score':cos,
                     'insight':preprocess(evaluation)})
        scores.append(cos)
        
    with open('result.json','w') as file:
        json.dump(data,file,indent=4)
    end = time.time()
    print(f'{end-start} sec')
    plot_histogram(scores)
    
if __name__ == '__main__':
    main()
    
