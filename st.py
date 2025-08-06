import fitz
import streamlit as st
import chromadb
from chromadb import EmbeddingFunction
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keybert import KeyBERT
import spacy
import matplotlib.pyplot as plt
import http.client
import json
import pandas as pd

# Streamlit App UI
st.title("🔍 Resume vs Job Description Analyzer")

# File Inputs
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")

if resume_file and job_desc_file:
    # st.success("Both files uploaded successfully!")

    api_key = st.secrets['GOOGLE_API_KEY']
    genai_client = genai.Client(api_key=api_key)

    resume = fitz.open(stream=resume_file.read(), filetype="pdf")
    job_desc = fitz.open(stream=job_desc_file.read(), filetype="pdf")

    cleaned_resume = ' '.join([page.get_text().lower().replace('\n', ' ') for page in resume])
    cleaned_job_desc = ' '.join([page.get_text().lower().replace('\n', ' ') for page in job_desc])

    # --- Embedding Function ---
    class GeminiEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):
            response = genai_client.models.embed_content(
                model='models/text-embedding-004',
                contents=input,
                config=types.EmbedContentConfig(task_type='retrieval_document')
            )
            return [e.values for e in response.embeddings]

    embed_fn = GeminiEmbeddingFunction()
    resume_vector = embed_fn(cleaned_resume)
    job_desc_vector = embed_fn(cleaned_job_desc)
    similarity_score = np.max(cosine_similarity(resume_vector, job_desc_vector)) * 100

    st.subheader("🔄 Similarity Score")
    st.metric("Resume to Job Match", f"{similarity_score:.2f}%")

    # --- Skill Matching ---
    st.subheader("🔍 Skill Matching")
    keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
    resume_skills = set([kw[0] for kw in keybert_model.extract_keywords(cleaned_resume, top_n=30)])
    job_desc_skills = set([kw[0] for kw in keybert_model.extract_keywords(cleaned_job_desc, top_n=30)])

    matched_skills = resume_skills & job_desc_skills
    missing_skills = job_desc_skills - resume_skills
    unique_skills = resume_skills - job_desc_skills

    st.write("**Matched Skills:**", matched_skills)
    st.write("**Missing (Required) Skills:**", missing_skills)
    st.write("**Extra Skills in Resume:**", unique_skills)

    # --- Interview Suggestions ---
    st.subheader("💬 Interview Tips")
    nlp = spacy.load('en_core_web_sm')
    sentences = [sent.text for sent in nlp(cleaned_job_desc).sents]

    responsibilities = [s for s in sentences if any(word in s.lower() for word in ['design', 'develop', 'maintain'])]
    requirements = [s for s in sentences if any(word in s.lower() for word in ['requirements', 'skills', 'experience'])]

    summary = genai_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=f"RESPONSIBILITIES: {responsibilities}\nREQUIREMENTS: {requirements}\nGenerate a 150-word summary."
    ).text

    st.markdown(summary)

    # --- Skill Match Pie Chart ---
    st.subheader("🌐 Visual Skill Match")
    match_score = (len(matched_skills) / (len(matched_skills) + len(missing_skills))) * 100
    fig, ax = plt.subplots()
    ax.pie([match_score, 100 - match_score], labels=["Matched", "Missing"], autopct="%1.1f%%")
    st.pyplot(fig)

    # --- Job Suggestions ---
    st.subheader("💼 Job Suggestions (via API)")
    conn = http.client.HTTPSConnection("jsearch.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': st.secrets['RAPIDAPI_KEY'],
        'x-rapidapi-host': "jsearch.p.rapidapi.com"
    }
    conn.request("GET", "/search?query=python%20developer%20karachi&page=1&num_pages=1&country=pk", headers=headers)
    res = conn.getresponse()
    job_data = json.loads(res.read().decode("utf-8"))
    jobs = job_data.get('data', [])

    if jobs:
        df = pd.DataFrame([{
            'Title': j['job_title'],
            'Company': j['employer_name'],
            'Type': j['job_employment_type'],
            'Location': j['job_location'],
            'Apply Link': j['job_apply_link'],
            'Description': j['job_description'][:200] + '...'
        } for j in jobs])
        st.dataframe(df)
    else:
        st.warning("No jobs found for 'Python Developer in Karachi'.")
