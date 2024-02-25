-- Using Gradio to wrap a text to text interface around GPT-2

!pip install gradio

-- Import, load model , generate function and create interface
  
import gradio as gr
from transformers import GPT2Tokenizer, TFAutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = TFAutoModelForCausalLM.from_pretrained("gpt2-medium")


def generate_text(message, history):
    input_ids = tokenizer.encode(message,return_tensors='tf')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5 , no_repeat_ngram_size=2, early_stopping=True)
    bot_response =tokenizer.decode(beam_output[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return bot_response


ui = gr.ChatInterface(generate_text,
                     title="CHATGPT-2",
                     description="OPEN AI's GPT-2 is an unsupervised language model that can generate coherent text. Go ahead and make a prompt in the textbox and check out what it generates.")
ui.launch()

