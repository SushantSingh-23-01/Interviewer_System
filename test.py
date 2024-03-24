import ollama, re, time, json

job_name = 'Data Scientist'
#Use gemma2b or other model for testing. 
#Use mistral or any better model for infernece
model = 'dolphin-mistral'

def interviewer_agent(job_name:str):

    prompt1 = f'''
    Imagine you are an interviewer hiring for a {job_name} position. You are supposed to ask one question. Do not provide answer. Conduct the interview in a profesional manner. Ask the question relevant to discussion. Take help of following context to form your question.
    '''

    prompt2 = f'''
    Imagine you are an interviewer. You are hiring  a {job_name}. Ask one question at a time. Keep the question short. Do not provide answer. 
    Come up with question based on the context given.
    '''
    prompt3 = f'''
    The following is an agent that is tasked at interviewing candidate for the job of {job_name}. The agent should ask 1 question at a time. It should refrain from giving any answers to the candidate. It should ask questions relevant to the context.
    Discussion:\n
    '''
    
    prompt4= f'''
    Act as an interviewer. You are interviewing user for the {job_name} position. Come up with question relevant to discussion.
    '''
    return prompt2

def llm_generate(model:str,prompt:str,user_input:str)->str:
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': f'{prompt}\n{user_input}',
    },
    ])
    result = str(response['message']['content'])
    result = result.replace('**', "") 
    return result

def preprocess(text):
    return re.sub(r"[^a-zA-Z?\"\']\*+",' ',text)

def write_txt(text):
    text_file = open("que_tts.txt", "w")
    text_file.write(text)
    text_file.close() 


def chat():
    # Not recommended to load text on memory.
    memory = []
    chat_log = ''
    print(f'You are going to be interviewed by an AI bot. Please introduce yourself to start the interview.')
    for i in range(0,10):     
        print('-'*100)
        user_input = input(f'\nYou: ')
        if user_input.lower() == 'quit':
            break
        elif len(chat_log) > 2048:
            chat_log = chat_log[-2048:]
        else: 
            pass
        
        start = time.time()
            
        question = llm_generate(
            model=model,
            prompt= interviewer_agent(job_name=job_name),
            user_input=user_input)
        print('-'*100)
        print(f'\nInterviewer: {question}\n')   
        end = time.time()
        print(f'{end-start:0.6f} seconds') 
                
        memory.append({'user':user_input,'interviewer':question})
        chat_log += f'\n{user_input}\n{question}'
        print('-'*100)
       
        
        #chat_log += 'User:\n' + user_input + 'Interviewer:\n' + question + '\n'
    with open('log.json','w') as file:
        json.dump(memory,file,indent=4) 
        
    
if __name__ == '__main__':
    chat()
