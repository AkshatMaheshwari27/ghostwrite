
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="GhostWrite",
    page_icon="👻",
    layout="centered"
)

# Custom theme
st.markdown("""
<style>
.stApp {
    background-color: #10111b;
    color: #e8e8f0;
}
h1, h2, h3 {
    color: #c4b5fd !important;
}
.stButton button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton button:hover {
    background-color: #6d28d9;
    color: white;
}
.story-box {
    background-color: #1c1d2b;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #4c1d95;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

st.title("👻 GhostWrite")
st.caption("Your imagination. Your ghost story. Your local AI.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

st.markdown("### 🕯️ Create your nightmare")

location = st.text_input(
    "📍 Setting",
    "An abandoned railway station"
)

character = st.text_input(
    "👤 Main character",
    "A night-shift engineer"
)

object_name = st.text_input(
    "📻 Mysterious object",
    "A broken radio"
)

length = st.select_slider(
    "📖 Story length",
    options=["Short", "Medium", "Long"],
    value="Medium"
)

if st.button("✨ Generate Ghost Story", use_container_width=True):

    word_limit = {
        "Short": 100,
        "Medium": 200,
        "Long": 300
    }

    prompt = f"""<|system|>
You are a creative horror writer. Write an original, atmospheric
ghost story. Use suspense, vivid sensory details, and a surprising
ending. Do not repeat the instructions.
</s>
<|user|>
Write a ghost story using these elements:
Setting: {location}
Main character: {character}
Mysterious object: {object_name}

Keep the story under {word_limit[length]} words.
</s>
<|assistant|>
"""

    with st.spinner("🌑 The spirits are writing..."):
        model = load_model()

        result = model(
            prompt,
            max_new_tokens=word_limit[length] * 3,
            do_sample=True,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.2,
            return_full_text=False,
            pad_token_id=2
        )

        story = result[0]["generated_text"].strip()

    st.markdown("### 📜 Your story")

    st.markdown(
        f'<div class="story-box">{story}</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download your story",
        data=story,
        file_name="ghostwrite_story.txt",
        mime="text/plain"
    )

st.divider()

st.caption(
    "GhostWrite • Open-source AI • Runs locally with TinyLlama"
)
