from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from gtts import gTTS
import os
import IPython

class ChatBot:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def chat(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
        chatbot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
        return response

# Example usage:
bot = ChatBot("currentlyexhausted/lite-llm")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chat ended.")
        break
    response = bot.chat(user_input)
    print("ChatBot:", response
    language = 'en'
    audio_file = gTTS(text=  response, lang=language, slow=False) 
    
    audio_file.save("audio.mp3")
    file = "./audio.mp3"
    IPython.display.display(IPython.display.Audio(file))

