import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any, Optional
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")

def log_feedback(event: Dict[str, Any]):
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def update_tag_affinity(state: Dict[str, Any], course: Course, signal: float):
    # Simple online mean-update toward tags of liked/high-rated courses
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