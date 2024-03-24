import subprocess
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import re
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer,util
from csv import DictWriter
from scrapper import scrape,parse_llm_output
import ollama

# embedding model
model_id = "sentence-transformers/all-MiniLM-L6-v2" # Context Length 384
embed_model = SentenceTransformer(model_id)

# Speech-to-text model
model_size = 'base.en'
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Transcribed audio location
audio_location = 'temp.wav'

######################################################

def preprocess(text):  
    text =  re.compile(r'<[^>]+>').sub('', text)
    text = re.sub(r'[^.,?a-zA-Z0-9]+',' ',text)  
    return text

def transcribe(audio_location):
    segments, _ = model.transcribe(audio_location)
    text = ''.join(segment.text for segment in segments)
    return text

def write_txt(text):
    text_file = open("que_tts.txt", "w")
    text_file.write(text)
    text_file.close()

def text_splitter(text,chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def chat_history(job,context,history):
    history = history[-4096:]
    response = ollama.chat(model='gemma:2b', messages=[
    {
        'role': 'user',
        'content': f'Act as a job interviewer. Interview the candidate for {job} position. Challenge the candidate to invoke critical and deep thinking. Take help of the context provided to form subsequent questions.\n{context}',
    },
    ])
    result = str(response['message']['content'])
    return result

def parse_llm_out(text):
    que_match = re.search(pattern = 'Question', string = text)
    parsed_que = text[que_match.end()+1:]
    return parsed_que
    
def similarity(user_ans, gen_ans):
    user_ans = embed_model.encode(user_ans)
    gen_ans = embed_model.encode(gen_ans)
    return util.cos_sim(user_ans,gen_ans).item()

def update_csv(output):
    field_names = ['gen_que','gen_ans','user_ans','cos_sim']
    with open('chatlog.csv', 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(output)
        f_object.close()

def main():     
    st.title('Interviewing System')       
    audio_location = 'user_rec.wav'
        
    audio_bytes = audio_recorder(pause_threshold=5.0)
    
    col1, col2, col3, col4 = st.columns([0.5,0.5,0.5,0.5])  
    
    if 'que' not in st.session_state:
        st.session_state.que = None
    if 'ans' not in st.session_state:
        st.session_state.ans = None
    if 'reply' not in st.session_state:
        st.session_state.reply = None
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
        
    with col1:
        if st.button('Generate Question'):    
            st.session_state.stage = 1
    
    if st.session_state.stage == 1:        
        qa_dict = scrape()  
        st.session_state.que = qa_dict.get('Question')
        st.session_state.ans = qa_dict.get('Answer')
        
        st.write(qa_dict.get('Question'))    
        
        write_txt(qa_dict.get('Question')) 
        
        subprocess.run(['python','audio.py'])
        st.session_state.stage = 2
    
    with col2:
        show2 = st.checkbox('Show Question')
        if show2:
            st.session_state.stage = 3
    
    if st.session_state.stage == 3:
        st.write(st.session_state.que)
        st.session_state.stage = 4
    
    if audio_bytes:   
        with open(audio_location,"wb") as f:
            f.write(audio_bytes)
        st.session_state.reply = transcribe(audio_location)
    
    with col3:
        show = st.checkbox('Show Transcribed Response')   
        if show:
            st.session_state.stage = 5
    
    if st.session_state.stage == 5:    
        if st.session_state.reply is not None:
            st.write(f'Answer: {st.session_state.reply}\n\n\n*Note: If transcribed answer does not match with answer you want to give. Do not press submit. Record Again and Check transcribed response.*') 
            st.session_state.stage = 6
            
    with col4:    
        if st.button('Submit Answer'):                                
            score = similarity(st.session_state.reply,st.session_state.que)       
            out_data = {
                "gen_que":st.session_state.que,
                "gen_ans":st.session_state.ans,
                "user_ans":st.session_state.reply,
                "cos_sim": score,
                }
            update_csv(out_data)
            st.session_state.stage = 0
            st.session_state.reply = None
            st.session_state.que = None
            st.session_state.ans = None
            
if __name__ == '__main__':
    main()

    
