import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    llm_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    model = AutoModelForCausalLM.from_pretrained(llm_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return embedder, tokenizer, model, device

embedder, tokenizer, llm_model, device = load_models()

# Data of carrer
career_paths = {
    "STEM": "Math, technology, engineering, coding, analytics",
    "Arts": "Creativity, design, painting, music, writing",
    "Sports": "Athletics, physical activity, games, teamwork, fitness",
    "Business": "Leadership, management, entrepreneurship, finance, marketing",
    "Healthcare": "Medicine, nursing, mental health, caregiving, diagnostics",
    "Social Sciences": "Psychology, sociology, human behavior, communication",
    "Law & Politics": "Debating, justice, legal systems, public speaking, policy-making",
    "Education": "Teaching, mentoring, training, learning methodologies",
    "Environment & Sustainability": "Ecology, climate change, conservation, sustainability",
    "Trades & Skilled Work": "Hands-on work, mechanical skills, craftsmanship, construction",
    "Media & Communication": "Journalism, content creation, social media, storytelling",
    "Gaming & Esports": "Competitive gaming, strategy, reflexes, digital collaboration",
    "Military & Defense": "Discipline, leadership, tactics, national service"
}

fallback_explanations = {
    "STEM": "You enjoy problem-solving and working with data or technology.",
    "Arts": "You have a creative mindset and a love for expression and design.",
    "Sports": "You show interest in physical activity, teamwork, and discipline.",
    "Business": "You're motivated by goals, leadership, and creating value through strategy.",
    "Healthcare": "You care deeply about helping others and are drawn to medical science.",
    "Social Sciences": "You’re curious about people, culture, and how society works.",
    "Law & Politics": "You’re passionate about justice, debate, and influencing change.",
    "Education": "You enjoy sharing knowledge and helping others learn and grow.",
    "Environment & Sustainability": "You’re driven to protect nature and build a greener future.",
    "Trades & Skilled Work": "You like hands-on work and solving real-world problems with skill.",
    "Media & Communication": "You thrive on storytelling, content, and connecting with people.",
    "Gaming & Esports": "You're strategic, competitive, and love digital challenges.",
    "Military & Defense": "You value discipline, honor, and service with a mission."
}

# App UI
st.title(" Career Companion")
st.markdown("Answer a few questions to discover a suitable career path based on your interests.")

responses = []
questions = [
    "What activities or topics do you enjoy the most in your free time?",
    "What kind of things excite or inspire you?",
    "What are your hobbies or things you naturally gravitate toward?",
]

with st.form("user_input_form"):
    for q in questions:
        responses.append(st.text_input(q))
    submitted = st.form_submit_button("Find My Career Path")

if submitted:
    user_input = " ".join(responses)
    user_input_normalized = re.sub(r"[^\w\s]", "", user_input.lower())
    ambiguous_keywords = ["don't know", "not sure", "no idea", "nothing", "none", "no interest"]

    if any(k in user_input_normalized for k in ambiguous_keywords):
        st.warning("It looks like you're not sure yet. Try answering these quick prompts instead:")
        extra_qs = [
            "Do you enjoy working with people, tools, computers, or ideas the most?",
            "What environment makes you feel most comfortable?",
            "What do you value more — creativity, logic, helping others, or leadership?",
            "If you could learn any one skill instantly, what would it be?",
            "What kind of problems do you like solving?"
        ]
        user_input = ""
        for q in extra_qs:
            user_input += st.text_input(q + " (quick prompt)", "") + " "

    user_embedding = embedder.encode(user_input, convert_to_tensor=True)
    scores = {
        path: util.cos_sim(user_embedding, embedder.encode(desc, convert_to_tensor=True)).item()
        for path, desc in career_paths.items()
    }
    best_match = max(scores, key=scores.get)

    st.success(f" Recommended Career Path: *{best_match}*")
    st.write(f" Why: {fallback_explanations[best_match]}")

    # Explaining why this carrer is good for the user  LLM  generation
    prompt = f"""
### Instruction:
A student described their interests as: "{user_input}"
Based on that, they were matched with the career path: "{best_match}".

Write a short paragraph (2–3 sentences) explaining why this career path fits the student's interests.
Use a helpful and neutral tone, as if an advisor is giving guidance.
Avoid first-person language like "I" or "me". Do not refer to yourself or the student directly.

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True).split("### Response:")[-1].strip()

    st.markdown(" *AI-Powered Insight:*")
    st.info(explanation)