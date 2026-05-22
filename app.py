import os
import gradio as gr
from huggingface_hub import InferenceClient

# Fetch the secret token safely from your Space Settings
hf_token = os.environ.get('HF_TOKEN')

# Initialize the free Llama-3 client
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)

SYSTEM_PROMPT = """
You are a friendly, clear, and empathetic medical assistant. Your goal is to answer general health-related questions.

CRITICAL SAFETY FILTERS:
1. You are an AI, NOT a doctor. 
2. If a user asks for a specific diagnosis, treatment plan, prescription dosage, or presents a life-threatening emergency, you MUST decline to give specific medical advice. Instead, gently redirect them to consult a qualified healthcare professional.
3. Always include a brief, non-intrusive legal medical disclaimer at the absolute end of your response where appropriate.
4. Keep the tone gentle, supportive, and informative. Do not use overly complex medical jargon without explaining it simply.
"""

def health_chatbot(user_message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Process history reliably across all Gradio versions (including v5 and v6)
    for interaction in history:
        if isinstance(interaction, dict):  # Modern dictionary message payload
            role = interaction.get("role")
            content = interaction.get("content")
            
            # Unpack structured OpenAI style blocks if present
            if isinstance(content, list):
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                content = " ".join(text_parts)
                
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": str(content)})
                
        elif isinstance(interaction, (list, tuple)) and len(interaction) == 2:  # Classic structural tuples
            user_val, ai_val = interaction
            if user_val:
                messages.append({"role": "user", "content": str(user_val)})
            if ai_val:
                messages.append({"role": "assistant", "content": str(ai_val)})
        
    # Append the newest user input query
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to backend model: {str(e)}. Please try again."

# Build UI Interface without passing theme here
demo = gr.ChatInterface(
    fn=health_chatbot,
    title="⚕️ General Health Query Chatbot",
    description="Ask general health questions. Powered by Llama-3-8B via Hugging Face's free inference layer.",
    examples=[
        "What causes a sore throat?", 
        "Is paracetamol safe for children?",
        "Can you prescribe me medicine for a severe chest pain?"
    ]
)

if __name__ == "__main__":
    # Theme configuration is passed here for Gradio 6 compatibility
    demo.launch(theme="soft")