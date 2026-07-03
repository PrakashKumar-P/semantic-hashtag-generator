import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import streamlit as st  
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
@st.cache_resource
def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()
@st.cache_data
def load_data():
        df = pd.read_csv(r"C:\Users\Prakash\OneDrive\Desktop\final for sub\trendsnap_large_dataset.csv")
        df['hashtags'] = df['hashtags'].fillna("").apply(
        lambda x: " ".join([tag.lstrip('#') for tag in str(x).split()])
    )
        captions = df['caption'].tolist()
        return df, captions
df, captions = load_data()
@st.cache_data
def get_caption_embeddings(captions):
        return model.encode(captions, convert_to_tensor=True)
caption_embeddings = get_caption_embeddings(captions)
def generate_hashtags(user_caption, top_k=3):
        if not user_caption.strip():
            return " Please enter a valid caption.", []
        user_embedding = model.encode(user_caption, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(user_embedding, caption_embeddings)[0]
        top_indices = np.argsort(-cosine_scores.cpu().numpy())[:top_k]
        hashtags = []
        for idx in top_indices:
            tags = df.iloc[idx]['hashtags'].split()
            hashtags.extend(tags)
        final_tags = list(dict.fromkeys(hashtags))[:50]
        return None, final_tags
st.set_page_config(page_title="TrendSnap (SBERT)", page_icon="📱")
st.title("#️⃣ TrendSnap: Hashtag Generator")
st.markdown("Type a caption below. We'll recommend relevant hashtags.")
user_input = st.text_area("✏️ Enter your caption here:", height=100, placeholder="e.g. Just baked a delicious chocolate cake ")

if st.button("Generate Hashtags"):
        with st.spinner("Thinking... 🧠"):
            error, hashtags = generate_hashtags(user_input)
        if error:
            st.warning(error)
        else:
            st.success("✅ Hashtags Generated:")
            st.markdown(" ".join([f"`#{tag}`" for tag in hashtags]))
