import fitz
from google import genai
from google.genai import types
import chromadb
from chromadb import EmbeddingFunction
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keybert import KeyBERT
import spacy
import streamlit as st
import plotly.graph_objects as go


st.markdown('# Resume Analysis')
genai_client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])

from google.api_core import retry
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

st.sidebar.header('Upload Documents')
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type="pdf")

def clean_doc(doc):
    doc = fitz.open(doc)
    pages_doc = [page.get_text().lower() for page in doc]
    cleaned_doc = ' '.join(pages_doc).replace('\n', ' ')
    return cleaned_doc

class GeminiEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input):
                response = genai_client.models.embed_content(
                    model='models/text-embedding-004',
                    contents=input,
                    config=types.EmbedContentConfig(task_type='retrieval_document')
                )
                return [e.values for e in response.embeddings]

keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

def convert_set(keywords):
    set_keywords = set([keyword[0] for keyword in keywords])
    return set_keywords

def extract_sentences(words):
    sentences = [s for s in words if any(word in s.lower() for word in words)]
    return sentences

if st.sidebar.button('Start Analysis'):
    if resume_file and job_desc_file:
        cleaned_resume = clean_doc(resume_file)
        cleaned_job_desc = clean_doc(job_desc_file)

        embed_fn = GeminiEmbeddingFunction()
        resume_vector = embed_fn(cleaned_resume)
        job_desc_vector = embed_fn(cleaned_job_desc)

        similarity = np.max(cosine_similarity(resume_vector, job_desc_vector)) * 100

        resume_skills = keybert_model.extract_keywords(cleaned_resume, top_n=30)
        job_desc_skills = keybert_model.extract_keywords(cleaned_job_desc, top_n=30)

        resume_skills = convert_set(resume_skills)
        job_desc_skills = convert_set(job_desc_skills)
        matched_skills = resume_skills.intersection(job_desc_skills)
        missing_skills = job_desc_skills - matched_skills
        unique_skills = resume_skills - matched_skills

        match_score = (len(resume_skills) / (len(resume_skills) + len(missing_skills))) * 100

        nlp = spacy.load('en_core_web_sm')
        job_desc_sentences = [sent.text for sent in nlp(cleaned_job_desc).sents]

        responsibility_words = ['responsibilities', 'design', 'develop', 'maintain', 'collaborate', 'write', 'optimize', 'review', 'integrate', 'implement']
        requirement_words = ['requirements', 'experience', 'degree', 'proficiency', 'familiarity', 'understanding', 'skills']

        responsibilities = extract_sentences(responsibility_words)
        requirements = extract_sentences(requirement_words)

        summary_job_desc = genai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"""
            You are an expert resume analyst. Given a list of job description sentences, generate a concise and professional summary (150–250 words) that
            outlines the key roles, responsibilities, and required skills.

            RESPONISIBLITIES: {responsibilities}
            REQUIREMENTS: {requirements}

            Now, generate the summary.
            """
        ).text

        summary_skills = genai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"""
            You are an expert resume analyst. Generate a concise and professional summary (150–250 words) that outlines candidate's strength and compatibility with given job description.

            Job Description: {cleaned_job_desc}
            Resume: {cleaned_resume}
            """
        ).text

        tab_names = ["🔍 Resume Match", "🧠 Skills Analysis", "📝 Job Summary", "Cover Letter"]

        if "active_tab" not in st.session_state or st.session_state.active_tab not in tab_names:
            st.session_state.active_tab = tab_names[0]

        tabs = st.tabs(tab_names)
        active_tab_index = tab_names.index(st.session_state.active_tab)

        with tabs[0]:
            st.session_state.active_tab = tab_names[0]
            st.markdown("## Similarity Between Resume and Job Description")
            st.write(f"{similarity:.2f}")
            st.markdown("## Resume Match Score")
            st.write(f"{match_score:.2f}")

        with tabs[1]:
            st.session_state.active_tab = tab_names[1]
            labels = ['Matched Skills', 'Missing Skills']
            values = [match_score, 100 - match_score]
            colors = ['lightgreen', 'lightcoral']
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.4)])
            fig.update_layout(title_text='Skill Match Analysis', title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            st.header('Matched Skills')
            for matched_skill in matched_skills:
                st.markdown(f"- {matched_skill.title()}")
            st.header('Missing Skills')
            for missing_skill in missing_skills:
                st.markdown(f"- {missing_skill.title()}")
            st.header('Summary of Skills')
            st.markdown(summary_skills)

        with tabs[2]:
            st.session_state.active_tab = tab_names[2]
            st.header('Summary of Job Description')
            st.markdown(summary_job_desc)

        with tabs[3]:
            st.session_state.active_tab = tab_names[3]
            st.header('Cover Letter for Job Description')
            cover_letter = genai_client.models.generate_content(
                model= 'gemini-2.0-flash',
                contents=f"""Write a professional and compelling cover letter in markdown format for the following job application.
                Requirements:
                1. Use **bold** text for section headings like "Introduction", "Skills & Experience", and "Closing".
                2. Use *italic* text to emphasize important phrases, tools, or achievements.
                3. Maintain a formal yet enthusiastic tone.
                4. Keep the letter concise (maximum 4 short paragraphs).
                5. Do not include personal contact details — only the body of the letter.
                JOB DESCRIPTION: {cleaned_job_desc}
                RESUME: {cleaned_resume}"""
            ).text
            st.markdown(cover_letter)

    else:
        st.warning('Upload Resume and Job Description First')
