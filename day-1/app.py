import streamlit as st
import google.generativeai as genai
import fitz
import requests
import plotly.express as px
import pandas as pd
import zipfile
import io
import re
import time
import json
from datetime import datetime

st.set_page_config(page_title="AI Research Paper Assistant", layout="wide", page_icon="📚")

CSS = """
<style>
.main { background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%); }
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1f2a44; }
.card {
    background: #ffffff; border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(31,42,68,0.07); border: 1px solid #eef0f5;
    margin-bottom: 1rem;
}
.gradient-header {
    background: linear-gradient(120deg, #4f6df5 0%, #7c5cff 100%);
    padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(79,109,245,0.25);
}
.gradient-header h1 { color: white; margin: 0; font-size: 1.7rem; }
.gradient-header p { color: #eef0ff; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.stButton>button {
    border-radius: 10px; font-weight: 600; border: none;
    background: linear-gradient(120deg, #4f6df5, #7c5cff); color: white;
    padding: 0.5rem 1.2rem; transition: all 0.2s;
}
.stButton>button:hover { box-shadow: 0 4px 14px rgba(79,109,245,0.4); transform: translateY(-1px); }
.pill {
    display: inline-block; background: #eef1ff; color: #4f6df5; padding: 0.2rem 0.7rem;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-right: 0.3rem;
}
.section-done { color: #1fae6e; font-weight: 600; }
.section-pending { color: #b3b8c5; }
[data-testid="stSidebar"] { background: #15192c; }
[data-testid="stSidebar"] * { color: #e7e9f5 !important; }
[data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #232a45 !important; color: #fff !important; border-radius: 8px;
}
hr { border-color: #e2e5ee; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DEFAULT_PROJECT = {
    "title": "", "domain": "", "keywords": [], "research_type": "",
    "journal": "IEEE", "citation_style": "IEEE", "authors": [], "papers": [],
    "topics": [], "selected_topic": None,
    "outline": "", "introduction": "", "literature_review": "",
    "lit_review_table": [], "methodology": "", "experiments": "",
    "exp_table": [], "results": "", "conclusion": "", "abstract": "",
    "references": [], "charts": []
}

if "project" not in st.session_state:
    st.session_state.project = json.loads(json.dumps(DEFAULT_PROJECT))
if "gemini_ready" not in st.session_state:
    st.session_state.gemini_ready = False

P = st.session_state.project

def init_gemini():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception:
        return None

MODEL = init_gemini()

def generate_text(prompt, max_tokens=1024, temperature=0.3, retries=2):
    if MODEL is None:
        st.error("Please add GOOGLE_API_KEY to Streamlit secrets.")
        return ""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = MODEL.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=max_tokens
                ),
            )
            if resp and resp.text:
                return resp.text.strip()
            last_err = "Empty response from model."
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
    st.warning(f"Generation failed: {last_err}")
    return ""

@st.cache_data(show_spinner=False)
def extract_pdf_metadata(file_bytes, filename):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            if len(full_text) > 6000:
                break
        meta = doc.metadata or {}
        doc.close()
        title = meta.get("title") or ""
        authors = meta.get("author") or ""
        if not title:
            lines = [l.strip() for l in full_text.split("\n") if l.strip()]
            title = lines[0] if lines else filename
        abstract = ""
        match = re.search(r"abstract[\s:]*\n?(.{50,1500}?)(?:\n\s*\n|introduction|keywords)", full_text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip().replace("\n", " ")
        year_match = re.search(r"(19|20)\d{2}", full_text[:3000])
        year = year_match.group(0) if year_match else "n.d."
        return {
            "filename": filename, "title": title.strip() or filename,
            "authors": authors.strip() or "Not available",
            "journal": "Not available", "year": year,
            "abstract": abstract or "Not available",
            "keywords": "Not available", "raw_text": full_text[:4000],
        }
    except Exception as e:
        return {
            "filename": filename, "title": filename, "authors": "Not available",
            "journal": "Not available", "year": "n.d.", "abstract": "Not available",
            "keywords": "Not available", "raw_text": "", "error": str(e),
        }

@st.cache_data(show_spinner=False)
def lookup_doi(doi):
    doi = doi.strip()
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        if r.status_code != 200:
            return None
        msg = r.json().get("message", {})
        title = msg.get("title", ["Not available"])
        authors_list = msg.get("author", [])
        authors = ", ".join(
            f"{a.get('given','')} {a.get('family','')}".strip() for a in authors_list
        ) or "Not available"
        journal = msg.get("container-title", ["Not available"])
        year = "n.d."
        for key in ("published-print", "published-online", "issued"):
            if key in msg and "date-parts" in msg[key]:
                parts = msg[key]["date-parts"][0]
                if parts:
                    year = str(parts[0])
                    break
        return {
            "filename": f"DOI:{doi}", "title": title[0] if title else "Not available",
            "authors": authors, "journal": journal[0] if journal else "Not available",
            "year": year, "abstract": "Not available", "keywords": "Not available",
            "publisher": msg.get("publisher", "Not available"), "doi": doi, "raw_text": "",
        }
    except Exception:
        return None

def progress_sections():
    keys = ["outline", "introduction", "literature_review", "methodology",
            "experiments", "results", "conclusion", "abstract"]
    labels = ["Outline", "Introduction", "Lit. Review", "Methodology",
              "Experiments", "Results", "Conclusion", "Abstract"]
    done = sum(1 for k in keys if P.get(k))
    return done, len(keys), list(zip(labels, keys))

def safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name) or "untitled"

def escape_latex(text):
    if not text:
        return ""
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text

def build_bibtex(papers):
    entries = []
    for i, p in enumerate(papers):
        key = safe_filename(p.get("title", f"ref{i}"))[:20] + str(p.get("year", ""))
        entries.append(
            f"@article{{{key},\n"
            f"  title={{{p.get('title','Not available')}}},\n"
            f"  author={{{p.get('authors','Not available')}}},\n"
            f"  journal={{{p.get('journal','Not available')}}},\n"
            f"  year={{{p.get('year','n.d.')}}}\n}}"
        )
    return "\n\n".join(entries)

def build_latex(project, paper_type):
    docclass = {
        "IEEE": r"\documentclass[conference]{IEEEtran}",
        "Springer": r"\documentclass{llncs}",
        "ACM": r"\documentclass[sigconf]{acmart}",
        "Elsevier": r"\documentclass[review]{elsarticle}",
    }.get(paper_type, r"\documentclass{article}")

    authors_tex = ", ".join(escape_latex(a.get("name", "")) for a in project["authors"]) or "Author Name"
    affil_tex = "; ".join(escape_latex(a.get("affiliation", "")) for a in project["authors"])

    body = f"""{docclass}
\\usepackage{{graphicx}}
\\usepackage{{cite}}
\\usepackage{{amsmath}}

\\title{{{escape_latex(project.get('title') or 'Untitled Research Paper')}}}
\\author{{{authors_tex}\\\\ {affil_tex}}}

\\begin{{document}}
\\maketitle

\\input{{sections/abstract.tex}}
\\input{{sections/introduction.tex}}
\\input{{sections/literature_review.tex}}
\\input{{sections/methodology.tex}}
\\input{{sections/results.tex}}
\\input{{sections/conclusion.tex}}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""
    return body

st.markdown(
    f"""<div class="gradient-header">
    <h1>📚 AI Research Paper Assistant</h1>
    <p>Guided, fact-grounded paper writing for first-time researchers — step by step.</p>
    </div>""",
    unsafe_allow_html=True,
)

if MODEL is None:
    st.error("Please add GOOGLE_API_KEY to Streamlit secrets.")

with st.sidebar:
    st.markdown("### 📋 Project Information")
    P["title"] = st.text_input("Project Title", value=P["title"])
    P["domain"] = st.text_input("Domain", value=P["domain"])
    kw_text = st.text_input("Keywords (comma separated)", value=", ".join(P["keywords"]))
    P["keywords"] = [k.strip() for k in kw_text.split(",") if k.strip()]
    P["research_type"] = st.selectbox(
        "Research Type", ["Survey", "Experimental", "Review", "Case Study", "Comparative Analysis"],
        index=["Survey", "Experimental", "Review", "Case Study", "Comparative Analysis"].index(P["research_type"]) if P["research_type"] else 0,
    )

    st.markdown("### 📄 Paper Format")
    P["journal"] = st.selectbox(
        "Paper Type", ["IEEE", "Springer", "ACM", "Elsevier"],
        index=["IEEE", "Springer", "ACM", "Elsevier"].index(P["journal"]) if P["journal"] in ["IEEE", "Springer", "ACM", "Elsevier"] else 0,
    )
    P["citation_style"] = st.selectbox(
        "Citation Style", ["IEEE", "APA", "MLA", "Chicago"],
        index=["IEEE", "APA", "MLA", "Chicago"].index(P["citation_style"]) if P["citation_style"] in ["IEEE", "APA", "MLA", "Chicago"] else 0,
    )

    st.markdown("### 👥 Authors")
    with st.expander("Add Author"):
        a_name = st.text_input("Name", key="a_name")
        a_aff = st.text_input("Affiliation", key="a_aff")
        a_desig = st.text_input("Designation", key="a_desig")
        a_email = st.text_input("Email", key="a_email")
        if st.button("Add Author", key="add_author_btn"):
            if a_name.strip():
                P["authors"].append({"name": a_name, "affiliation": a_aff, "designation": a_desig, "email": a_email})
                st.success(f"Added {a_name}")
                st.rerun()
            else:
                st.warning("Name is required.")
    for i, a in enumerate(P["authors"]):
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{a['name']}** — {a.get('affiliation','')}")
        if c2.button("✕", key=f"rm_author_{i}"):
            P["authors"].pop(i)
            st.rerun()

    st.markdown("### 📈 Progress")
    done, total, sec_list = progress_sections()
    st.progress(done / total if total else 0)
    st.caption(f"{done} / {total} sections complete")
    for label, key in sec_list:
        if P.get(key):
            st.markdown(f"<span class='section-done'>✅ {label}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='section-pending'>⬜ {label}</span>", unsafe_allow_html=True)

tabs = st.tabs([
    "1️⃣ Topic", "2️⃣ References", "3️⃣ Lit. Review", "4️⃣ Outline", "5️⃣ Introduction",
    "6️⃣ Methodology", "7️⃣ Experiments", "8️⃣ Results", "9️⃣ Conclusion", "🔟 Abstract", "📦 Export"
])

with tabs[0]:
    st.subheader("Topic Generator")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        t_domain = st.text_input("Domain", value=P["domain"], key="topic_domain")
        t_keywords = st.text_input("Keywords", value=", ".join(P["keywords"]), key="topic_keywords")
    with col2:
        t_difficulty = st.select_slider("Difficulty", ["Beginner", "Intermediate", "Advanced"], key="topic_difficulty")
        t_rtype = st.selectbox("Research Type", ["Survey", "Experimental", "Review", "Case Study", "Comparative Analysis"], key="topic_rtype")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✨ Generate 5 Topic Ideas"):
        if not t_domain.strip():
            st.warning("Please enter a domain.")
        else:
            with st.spinner("Generating topic ideas..."):
                prompt = (
                    f"Suggest exactly 5 research paper topics for a first-time college researcher.\n"
                    f"Domain: {t_domain}\nKeywords: {t_keywords}\nDifficulty: {t_difficulty}\n"
                    f"Research Type: {t_rtype}\n\n"
                    "Respond ONLY with valid JSON, an array of 5 objects, each with keys: "
                    "title, description, novelty_score (1-10), difficulty_score (1-10), research_gap. "
                    "No markdown, no commentary, no code fences."
                )
                raw = generate_text(prompt, max_tokens=1200)
                raw_clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                try:
                    topics = json.loads(raw_clean)
                    P["topics"] = topics
                except Exception:
                    st.error("Could not parse topic ideas. Please try again.")
                    P["topics"] = []

    if P["topics"]:
        for i, t in enumerate(P["topics"]):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{t.get('title','Untitled')}**")
            st.write(t.get("description", ""))
            cc1, cc2, cc3 = st.columns(3)
            cc1.markdown(f"<span class='pill'>Novelty: {t.get('novelty_score','-')}/10</span>", unsafe_allow_html=True)
            cc2.markdown(f"<span class='pill'>Difficulty: {t.get('difficulty_score','-')}/10</span>", unsafe_allow_html=True)
            cc3.write("")
            st.caption(f"Research Gap: {t.get('research_gap','')}")
            if st.button("Select this topic", key=f"select_topic_{i}"):
                P["selected_topic"] = t
                P["title"] = t.get("title", P["title"])
                st.success("Topic selected and saved as project title.")
            st.markdown("</div>", unsafe_allow_html=True)

    if P["selected_topic"]:
        st.info(f"✅ Selected Topic: **{P['selected_topic'].get('title')}**")

with tabs[1]:
    st.subheader("Upload Reference PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        existing_names = {p["filename"] for p in P["papers"]}
        for f in uploaded_files:
            if f.name not in existing_names:
                with st.spinner(f"Extracting {f.name}..."):
                    meta = extract_pdf_metadata(f.getvalue(), f.name)
                    P["papers"].append(meta)
        st.success(f"Processed {len(uploaded_files)} file(s).")

    st.markdown("#### DOI Lookup")
    dcol1, dcol2 = st.columns([3, 1])
    doi_input = dcol1.text_input("Enter DOI (e.g. 10.1000/xyz123)", key="doi_input")
    if dcol2.button("Lookup DOI"):
        if doi_input.strip():
            with st.spinner("Querying Crossref..."):
                result = lookup_doi(doi_input.strip())
            if result:
                P["papers"].append(result)
                st.success(f"Added: {result['title']}")
            else:
                st.warning("Invalid DOI or not found in Crossref.")
        else:
            st.warning("Please enter a DOI.")

    st.markdown("#### Reference Papers")
    if not P["papers"]:
        st.info("No papers added yet. Upload PDFs or look up a DOI above.")
    else:
        df_view = pd.DataFrame([
            {"Title": p.get("title", ""), "Authors": p.get("authors", ""),
             "Year": p.get("year", ""), "Journal": p.get("journal", "")}
            for p in P["papers"]
        ])
        st.dataframe(df_view, use_container_width=True)
        for i, p in enumerate(P["papers"]):
            with st.expander(f"📄 {p.get('title','Untitled')}"):
                st.write(f"**Authors:** {p.get('authors','Not available')}")
                st.write(f"**Year:** {p.get('year','Not available')}")
                st.write(f"**Journal:** {p.get('journal','Not available')}")
                st.write(f"**Abstract:** {p.get('abstract','Not available')}")
                if st.button("Remove", key=f"rm_paper_{i}"):
                    P["papers"].pop(i)
                    st.rerun()

with tabs[2]:
    st.subheader("Literature Review")
    if not P["papers"]:
        st.warning("Add reference papers in Tab 2 first.")
    else:
        if st.button("📝 Generate Literature Review"):
            with st.spinner("Analyzing uploaded papers..."):
                papers_summary = "\n\n".join(
                    f"Title: {p.get('title')}\nAuthors: {p.get('authors')}\nYear: {p.get('year')}\n"
                    f"Journal: {p.get('journal')}\nAbstract: {p.get('abstract')}"
                    for p in P["papers"]
                )
                prompt = (
                    "You are writing the Literature Review section of an academic paper. "
                    "Use ONLY the paper information supplied below. Do not invent facts, "
                    "authors, datasets, or results. If a detail is unavailable, write 'Not available'.\n\n"
                    f"PAPERS:\n{papers_summary}\n\n"
                    "Write a well-structured literature review (3-5 paragraphs) synthesizing these papers, "
                    "their relationships, and gaps."
                )
                P["literature_review"] = generate_text(prompt, max_tokens=1200)

                table_prompt = (
                    "Based ONLY on the papers below, produce a JSON array. Each object has keys: "
                    "paper (title), method, dataset, result, limitation. Use 'Not available' if unknown. "
                    "No commentary, no markdown.\n\n" + papers_summary
                )
                raw_table = generate_text(table_prompt, max_tokens=1000)
                raw_table_clean = re.sub(r"^```json|```$", "", raw_table.strip(), flags=re.MULTILINE).strip()
                try:
                    P["lit_review_table"] = json.loads(raw_table_clean)
                except Exception:
                    P["lit_review_table"] = []

        if P["literature_review"]:
            P["literature_review"] = st.text_area("Literature Review", value=P["literature_review"], height=280)
            st.markdown("#### Comparison Table")
            if P["lit_review_table"]:
                lit_df = pd.DataFrame(P["lit_review_table"])
                edited = st.data_editor(lit_df, use_container_width=True, num_rows="dynamic")
                P["lit_review_table"] = edited.to_dict("records")

with tabs[3]:
    st.subheader("Outline")
    if st.button("🗂️ Generate Outline"):
        with st.spinner("Building outline..."):
            prompt = (
                f"Generate a numbered research paper outline for the title '{P['title'] or 'Untitled'}' "
                f"in the domain {P['domain']}, research type {P['research_type']}. "
                "Use exactly these top-level sections in order: 1 Abstract, 2 Introduction, "
                "3 Literature Review, 4 Methodology, 5 Experiments, 6 Results, 7 Discussion, "
                "8 Conclusion, 9 Future Work, 10 References. Add 2-3 brief sub-points under each."
            )
            P["outline"] = generate_text(prompt, max_tokens=900)
    if P["outline"]:
        P["outline"] = st.text_area("Outline", value=P["outline"], height=400)

with tabs[4]:
    st.subheader("Introduction")
    if st.button("✍️ Generate Introduction"):
        with st.spinner("Writing introduction..."):
            papers_summary = "\n".join(f"- {p.get('title')}: {p.get('abstract')}" for p in P["papers"]) or "No papers uploaded."
            gap = P["selected_topic"].get("research_gap", "") if P["selected_topic"] else ""
            prompt = (
                f"Write an academic Introduction section for the paper titled '{P['title'] or 'Untitled'}'.\n"
                f"Keywords: {', '.join(P['keywords'])}\nResearch Gap: {gap}\n"
                f"Reference paper summaries:\n{papers_summary}\n\n"
                "Cover: Background, Problem Statement, Research Gap, Objectives, Contributions, "
                "and Paper Organization, in flowing academic prose with clear paragraph breaks. "
                "Do not invent citations or facts not given above."
            )
            P["introduction"] = generate_text(prompt, max_tokens=1200)
    if P["introduction"]:
        P["introduction"] = st.text_area("Introduction", value=P["introduction"], height=400)

with tabs[5]:
    st.subheader("Methodology")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    m_design = st.text_input("Research Design", key="m_design")
    m_dataset = st.text_input("Dataset", key="m_dataset")
    m_models = st.text_input("Models", key="m_models")
    m_metrics = st.text_input("Evaluation Metrics", key="m_metrics")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⚙️ Generate Methodology"):
        with st.spinner("Writing methodology..."):
            prompt = (
                "Write an academic Methodology section using strictly the details below. "
                "Do not invent datasets, models, or metrics beyond what is given. "
                "Explain the rationale, pipeline, and evaluation approach clearly.\n\n"
                f"Research Design: {m_design}\nDataset: {m_dataset}\n"
                f"Models: {m_models}\nEvaluation Metrics: {m_metrics}"
            )
            P["methodology"] = generate_text(prompt, max_tokens=1000)
    if P["methodology"]:
        P["methodology"] = st.text_area("Methodology", value=P["methodology"], height=350)

with tabs[6]:
    st.subheader("Experiments")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    e_dataset = st.text_input("Dataset", key="e_dataset")
    e_models = st.text_input("Models", key="e_models")
    e_epochs = st.text_input("Epochs", key="e_epochs")
    e_optimizer = st.text_input("Optimizer", key="e_optimizer")
    e_details = st.text_area("Training Details", key="e_details")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🧪 Generate Experiment Description"):
        with st.spinner("Writing experiments section..."):
            prompt = (
                "Write an academic Experiments section describing the setup below. "
                "Do not invent performance numbers; describe only the setup process.\n\n"
                f"Dataset: {e_dataset}\nModels: {e_models}\nEpochs: {e_epochs}\n"
                f"Optimizer: {e_optimizer}\nDetails: {e_details}"
            )
            P["experiments"] = generate_text(prompt, max_tokens=900)
    if P["experiments"]:
        P["experiments"] = st.text_area("Experiment Description", value=P["experiments"], height=300)

    st.markdown("#### Results Table")
    if not P["exp_table"]:
        P["exp_table"] = [{"Model": "", "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}]
    exp_df = pd.DataFrame(P["exp_table"])
    edited_exp = st.data_editor(exp_df, use_container_width=True, num_rows="dynamic", key="exp_editor")
    P["exp_table"] = edited_exp.to_dict("records")

    st.markdown("#### Chart Generator")
    if P["exp_table"] and any(r.get("Model") for r in P["exp_table"]):
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie"], key="chart_type")
        metric = st.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1"], key="chart_metric")
        chart_df = pd.DataFrame(P["exp_table"])
        try:
            if chart_type == "Bar":
                fig = px.bar(chart_df, x="Model", y=metric, title=f"{metric} by Model")
            elif chart_type == "Line":
                fig = px.line(chart_df, x="Model", y=metric, title=f"{metric} by Model")
            else:
                fig = px.pie(chart_df, names="Model", values=metric, title=f"{metric} Distribution")
            st.plotly_chart(fig, use_container_width=True)
            png_bytes = fig.to_image(format="png")
            st.download_button("Download Chart PNG", png_bytes, file_name=f"{metric}_{chart_type}.png")
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

with tabs[7]:
    st.subheader("Results & Discussion")
    if st.button("📊 Generate Results Discussion"):
        if not P["exp_table"] or not any(r.get("Model") for r in P["exp_table"]):
            st.warning("Add experiment results in Tab 7 first.")
        else:
            with st.spinner("Analyzing results..."):
                table_str = pd.DataFrame(P["exp_table"]).to_string(index=False)
                prompt = (
                    "Write a Results and Discussion section using ONLY the table data below. "
                    "Do not invent numbers not present in the table.\n\n"
                    f"{table_str}\n\n"
                    "Cover: Observations, Strengths, Weaknesses, Future Improvements."
                )
                P["results"] = generate_text(prompt, max_tokens=1000)
    if P["results"]:
        P["results"] = st.text_area("Results & Discussion", value=P["results"], height=350)

with tabs[8]:
    st.subheader("Conclusion")
    if st.button("🏁 Generate Conclusion"):
        with st.spinner("Writing conclusion..."):
            prompt = (
                f"Write a Conclusion section for the paper '{P['title'] or 'Untitled'}' based on:\n"
                f"Introduction: {P['introduction'][:800]}\n"
                f"Methodology: {P['methodology'][:800]}\n"
                f"Results: {P['results'][:800]}\n\n"
                "Cover: Summary, Contributions, Limitations, Future Work."
            )
            P["conclusion"] = generate_text(prompt, max_tokens=800)
    if P["conclusion"]:
        P["conclusion"] = st.text_area("Conclusion", value=P["conclusion"], height=300)

with tabs[9]:
    st.subheader("Abstract")
    st.caption("Generate this last, once other sections are complete.")
    if st.button("📌 Generate Abstract"):
        with st.spinner("Synthesizing abstract..."):
            prompt = (
                "Write a concise academic abstract (150-250 words) summarizing the following paper sections. "
                "Do not introduce new facts.\n\n"
                f"Introduction: {P['introduction'][:600]}\n"
                f"Literature Review: {P['literature_review'][:600]}\n"
                f"Methodology: {P['methodology'][:600]}\n"
                f"Results: {P['results'][:600]}\n"
                f"Conclusion: {P['conclusion'][:600]}"
            )
            P["abstract"] = generate_text(prompt, max_tokens=400)
    if P["abstract"]:
        P["abstract"] = st.text_area("Abstract", value=P["abstract"], height=220)
        wc = len(P["abstract"].split())
        cc = len(P["abstract"])
        st.caption(f"Word Count: {wc} | Character Count: {cc}")
        if wc < 150 or wc > 250:
            st.warning("Recommended length is 150-250 words.")

with tabs[10]:
    st.subheader("Export to LaTeX (Overleaf-ready)")
    st.write(f"Paper Type: **{P['journal']}** | Citation Style: **{P['citation_style']}**")

    missing = [k for k in ["introduction", "literature_review", "methodology", "results", "conclusion", "abstract"] if not P.get(k)]
    if missing:
        st.warning(f"These sections are still empty: {', '.join(missing)}. You can still export a partial draft.")

    if st.button("📦 Build Project ZIP"):
        with st.spinner("Compiling LaTeX project..."):
            try:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("main.tex", build_latex(P, P["journal"]))
                    zf.writestr("references.bib", build_bibtex(P["papers"]) or "% No references added")
                    zf.writestr("sections/abstract.tex", "\\begin{abstract}\n" + escape_latex(P["abstract"] or "Not available") + "\n\\end{abstract}")
                    zf.writestr("sections/introduction.tex", "\\section{Introduction}\n" + escape_latex(P["introduction"] or "Not available"))
                    zf.writestr("sections/literature_review.tex", "\\section{Literature Review}\n" + escape_latex(P["literature_review"] or "Not available"))
                    zf.writestr("sections/methodology.tex", "\\section{Methodology}\n" + escape_latex(P["methodology"] or "Not available"))
                    zf.writestr("sections/results.tex", "\\section{Results and Discussion}\n" + escape_latex(P["results"] or "Not available"))
                    zf.writestr("sections/conclusion.tex", "\\section{Conclusion}\n" + escape_latex(P["conclusion"] or "Not available"))
                    zf.writestr("images/.gitkeep", "")
                buf.seek(0)
                st.session_state["export_zip"] = buf.getvalue()
                st.success("Project ZIP built successfully.")
            except Exception as e:
                st.error(f"Export failed: {e}")

    if "export_zip" in st.session_state:
        st.download_button(
            "⬇️ Download project.zip",
            data=st.session_state["export_zip"],
            file_name=f"{safe_filename(P['title'] or 'research_project')}.zip",
            mime="application/zip",
        )