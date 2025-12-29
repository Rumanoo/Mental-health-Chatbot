import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Empathetic Support Bot", page_icon="🌱")
st.title("🌱 Mental Health Support Chatbot")
st.markdown("I am a fine-tuned AI assistant trained to respond with empathy.")

@st.cache_resource
def load_model():
    
    model_path = "Rumanoo/mental-health-chatbot" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tell me how you're feeling..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_prompt = f"User: {prompt}\nAssistant (empathetic response):"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        final_text = response.split("Assistant (empathetic response):")[-1].strip()
        
        st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})

st.divider()
st.caption("⚠️ Disclaimer: This bot is an AI research project and not a licensed professional.")
