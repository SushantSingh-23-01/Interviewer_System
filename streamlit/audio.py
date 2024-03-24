import pyttsx3

txt_file = 'que_tts.txt'
with open(txt_file) as f:
    TEXT = f.read()
 
# init function to get an engine instance for the speech synthesis 
engine = pyttsx3.init()
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[1].id)

rate = engine.getProperty('rate')   # getting details of current speaking rate                     
engine.setProperty('rate', 150)     # setting up new voice rate
 
# say method on the engine that passing input text to be spoken
engine.say(TEXT)
 
# run and wait method, it processes the voice commands. 
engine.runAndWait()
