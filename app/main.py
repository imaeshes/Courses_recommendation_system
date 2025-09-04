from __future__ import annotations
import os,sys
import json
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.template import SYSTEM_PROMPT
from rag import build_context, openai_chat


 
# Data models
 
class Course(BaseModel):
    title: str
    provider: str
    description: str
    skill_level: str  # e.g., beginner, intermediate, advanced
    tags: List[str]

class UserProfile(BaseModel):
    background: str
    interests: List[str]
    goals: str
    skills: Dict[str, int] = Field(default_factory=dict)  # e.g., {"programming":3, "math":2}

 
# Paths
 
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")
PROFILE_STATE_PATH = os.path.join(DATA_DIR, "profile_state.json")

 
# Dataset loading
 


CSV_PATH = os.path.join(os.getcwd(), "data/courses.csv")


def load_courses() -> List[Course]:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = df.fillna("")
        records = []
        for _, r in df.iterrows():
            tags = [t.strip() for t in str(r.get("tags", "")).split(",") if t.strip()]
            records.append(Course(
                title=str(r.get("title")),
                provider=str(r.get("provider")),
                description=str(r.get("description")),
                skill_level=str(r.get("skill_level")),
                tags=tags,
            ))
        return records


 
# Embeddings + Vector Index
 
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def build_index(courses: List[Course], model) -> tuple[faiss.IndexFlatIP, np.ndarray, List[str]]:
    # We'll embed concatenated text fields for each course
    texts = [f"{c.title}. {c.provider}. {c.description}. Tags: {', '.join(c.tags)}. Level: {c.skill_level}." for c in courses]
    embs = model.encode(texts, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via normalized dot product
    index.add(embs.astype(np.float32))
    return index, embs, texts

 
# Personalization state
 
DEFAULT_PROFILE_STATE = {
    "tag_affinity": {},   # tag -> float score
    "likes": {},          # course_title -> count
    "ratings": {},        # course_title -> avg_rating
}


def load_profile_state() -> Dict[str, Any]:
    if os.path.exists(PROFILE_STATE_PATH):
        try:
            with open(PROFILE_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_PROFILE_STATE.copy()


def save_profile_state(state: Dict[str, Any]):
    with open(PROFILE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

 
# Scoring with personalization
 

def compute_tag_boost(course: Course, tag_affinity: Dict[str, float]) -> float:
    if not tag_affinity:
        return 0.0
    scores = [tag_affinity.get(t, 0.0) for t in course.tags]
    if not scores:
        return 0.0
    return float(np.mean(scores))


def rank_courses(user_text: str, courses: List[Course], index, embedder, tag_affinity: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
    q_emb = embedder.encode([user_text], normalize_embeddings=True).astype(np.float32)
    sims, ids = index.search(q_emb, min(top_k*4, len(courses)))  # overfetch then re-rank
    sims = sims[0]
    ids = ids[0]

    # Blend similarity with tag affinity
    alpha = 0.75  # content similarity weight
    beta = 0.25   # personalization weight
    ranked = []
    for sim, idx in zip(sims, ids):
        c = courses[idx]
        boost = compute_tag_boost(c, tag_affinity)
        score = alpha * float(sim) + beta * boost
        ranked.append({"course": c, "sim": float(sim), "boost": boost, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

 
# Feedback persistence + learning

# using TAG AFFINITY MODELLING 

def log_feedback(event: Dict[str, Any]):
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def update_tag_affinity(state: Dict[str, Any], course: Course, signal: float):
    """
    Called whenever user gives a rating (1–5 stars).
    Neutral = 3. Anything above → positive reinforcement; below → negative.
    Formula:
    (rating - 3) * 0.5
    Rating 5 → +1.0 boost per tag.
    Rating 1 → -1.0 penalty per tag.
    Why 0.5 scale?
    Keeps weights modest → avoids overfitting to one click.
    Gradual learning effect across sessions.
    Updates stored in PROFILE_STATE["tag_affinity"].
    Example: If user loves ML courses (rating 5), "machine-learning" tag weight increases, making future ML courses rank higher.
    """
    tag_aff = state.get("tag_affinity", {})
    for t in course.tags:
        prev = tag_aff.get(t, 0.0)
        # bounded update
        new_val = float(np.clip(prev + 0.1 * signal, -1.0, 1.0))
        tag_aff[t] = new_val
    state["tag_affinity"] = tag_aff


def update_course_rating(state: Dict[str, Any], title: str, rating: float):
    ratings = state.get("ratings", {})
    prev = ratings.get(title, None)
    if prev is None:
        ratings[title] = float(rating)
    else:
        # Moving average with small inertia
        ratings[title] = float(0.8 * prev + 0.2 * rating)
    state["ratings"] = ratings


 
# UI
 
st.set_page_config(page_title="Course & Career Recommender", page_icon="🎓", layout="wide")
st.title("🎓 Intelligent Course & Career Path Recommender")

with st.expander("About this app", expanded=False):
    st.markdown(
        "This mini app uses sentence embeddings for course matching, offers retrieval-based Q&A over the catalog, and learns from your feedback to personalize future recommendations."
    )

# Load data and models
courses = load_courses()
embedder = get_embedder()
index, embs, texts = build_index(courses, embedder)
profile_state = load_profile_state()

# Sidebar: profile input
st.sidebar.header("Your Profile")
background = st.sidebar.text_input("Background", placeholder="Final-year CS student")
interests_text = st.sidebar.text_input("Interests (comma-separated)", placeholder="AI, Data Science, Backend")
goals = st.sidebar.text_area("Career Goals", placeholder="Become an ML Engineer at a product company")

st.sidebar.subheader("Self-rated skills (0-5)")
skill_names = ["programming", "mathematics", "statistics", "ml", "data-engineering", "cloud", "frontend", "backend"]
skills: Dict[str, int] = {}
cols = st.sidebar.columns(2)
for i, name in enumerate(skill_names):
    with cols[i % 2]:
        skills[name] = int(st.slider(name.capitalize(), 0, 5, 3))

profile = UserProfile(
    background=background.strip(),
    interests=[s.strip() for s in interests_text.split(",") if s.strip()],
    goals=goals.strip(),
    skills=skills,
)

# Generate profile text for embedding
profile_text_parts = []
if profile.background:
    profile_text_parts.append(f"Background: {profile.background}.")
if profile.interests:
    profile_text_parts.append(f"Interests: {', '.join(profile.interests)}.")
if profile.goals:
    profile_text_parts.append(f"Goals: {profile.goals}.")
if profile.skills:
    skills_str = ", ".join([f"{k}:{v}" for k,v in profile.skills.items()])
    profile_text_parts.append(f"Skills: {skills_str}.")
profile_text = " ".join(profile_text_parts) or "Aspiring technologist."

# Recommend
st.header("🔍 Recommendations")
with st.spinner("Matching courses to your profile..."):
    ranked = rank_courses(profile_text, courses, index, embedder, profile_state.get("tag_affinity", {}), top_k=5)

for i, item in enumerate(ranked, start=1):
    c: Course = item["course"]
    with st.container(border=True):
        st.markdown(f"**#{i}. {c.title}**  ")
        st.caption(f"Provider: {c.provider} • Level: {c.skill_level} • Tags: {', '.join(c.tags)}")
        st.write(c.description)
        st.progress(min(max((item['score']+1)/2, 0.0), 1.0), text=f"Match score: {item['score']:.2f} (sim {item['sim']:.2f} + boost {item['boost']:.2f})")

        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("👍 Like", key=f"like_{i}"):
                # positive signal
                update_tag_affinity(profile_state, c, signal=+1.0)
                log_feedback({
                    "ts": time.time(),
                    "event": "like",
                    "course": c.dict(),
                    "profile_text": profile_text,
                })
                save_profile_state(profile_state)
                st.success("Thanks! We'll personalize future picks.")
        with cols[1]:
            if st.button("👎 Skip", key=f"skip_{i}"):
                update_tag_affinity(profile_state, c, signal=-0.5)
                log_feedback({
                    "ts": time.time(),
                    "event": "skip",
                    "course": c.dict(),
                    "profile_text": profile_text,
                })
                save_profile_state(profile_state)
                st.info("Got it. We'll show fewer like this.")
        with cols[2]:
            rating = st.slider("Rate this recommendation", 1, 5, 4, key=f"rate_{i}")
            if st.button("Submit rating", key=f"rate_btn_{i}"):
                update_course_rating(profile_state, c.title, float(rating))
                update_tag_affinity(profile_state, c, signal=(rating-3)/2)
                log_feedback({
                    "ts": time.time(),
                    "event": "rating",
                    "rating": float(rating),
                    "course_title": c.title,
                    "profile_text": profile_text,
                })
                save_profile_state(profile_state)
                st.success("Rating saved!")

# QA / RAG
st.header("💬 Ask questions about the catalog (RAG)")
user_q = st.text_input("Your question", placeholder="Which beginner-friendly courses focus on Python and data analysis?")
rag_cols = st.columns([2,1])
with rag_cols[0]:
    if st.button("Answer with RAG") and user_q.strip():
        with st.spinner("Retrieving and answering..."):
            ctx_snippets = build_context(user_q, texts, embedder, k=6)
            prompt = (
                SYSTEM_PROMPT
                + "\n\nContext:\n" + "\n\n".join([f"- {c}" for c in ctx_snippets])
                + "\n\nUser question: " + user_q
            )
            answer = openai_chat([
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":prompt},
            ])
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Context used"):
                for s in ctx_snippets:
                    st.markdown(f"- {s}")
            log_feedback({
                "ts": time.time(),
                "event": "rag_q",
                "question": user_q,
                "context_count": len(ctx_snippets),
            })

with rag_cols[1]:
    st.markdown("**Tips**")
    st.markdown("- Ask about levels, providers, technologies, or learning paths.\n- Use ratings/likes to tune future picks.")



