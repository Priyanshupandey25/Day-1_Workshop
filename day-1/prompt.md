import os
# Mandatory for Vercel/Serverless: Matplotlib needs a writable config directory
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for serverless
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from google import genai
from google.genai import types
import fitz  # PyMuPDF
import requests
import pandas as pd
import zipfile
import io
import re
import time
import json
import base64

app = Flask(__name__)

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def generate_text(prompt, max_tokens=1024, temperature=0.3, retries=2):
    client = get_gemini_client()
    if not client:
        return "Error: GEMINI_API_KEY environment variable is not set."
    
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            if response and response.text:
                return response.text.strip()
            last_err = "Empty response from model."
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
    return f"Generation failed: {last_err}"

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

    authors_tex = ", ".join(escape_latex(a.get("name", "")) for a in project.get("authors", [])) or "Author Name"
    affil_tex = "; ".join(escape_latex(a.get("affiliation", "")) for a in project.get("authors", []))

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

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate_topics", methods=["POST"])
def api_generate_topics():
    data = request.json
    prompt = (
        f"Suggest exactly 5 research paper topics for a first-time college researcher.\n"
        f"Domain: {data.get('domain')}\nKeywords: {data.get('keywords')}\nDifficulty: {data.get('difficulty')}\n"
        f"Research Type: {data.get('rtype')}\n\n"
        "Respond ONLY with valid JSON, an array of 5 objects, each with keys: "
        "title, description, novelty_score (1-10), difficulty_score (1-10), research_gap. "
        "No markdown, no commentary, no code fences."
    )
    raw = generate_text(prompt, max_tokens=1200)
    raw_clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        topics = json.loads(raw_clean)
        return jsonify({"success": True, "topics": topics})
    except Exception as e:
        return jsonify({"success": False, "error": "Could not parse topic ideas."})

@app.route("/api/extract_pdf", methods=["POST"])
def api_extract_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    file = request.files["file"]
    meta = extract_pdf_metadata(file.read(), file.filename)
    return jsonify({"success": True, "paper": meta})

@app.route("/api/lookup_doi", methods=["POST"])
def api_lookup_doi():
    doi = request.json.get("doi")
    result = lookup_doi(doi)
    if result:
        return jsonify({"success": True, "paper": result})
    return jsonify({"success": False, "error": "Invalid DOI or not found in Crossref."})

@app.route("/api/generate_lit_review", methods=["POST"])
def api_generate_lit_review():
    papers = request.json.get("papers", [])
    papers_summary = "\n\n".join(
        f"Title: {p.get('title')}\nAuthors: {p.get('authors')}\nYear: {p.get('year')}\n"
        f"Journal: {p.get('journal')}\nAbstract: {p.get('abstract')}"
        for p in papers
    )
    prompt_text = (
        "You are writing the Literature Review section of an academic paper. "
        "Use ONLY the paper information supplied below. Do not invent facts, "
        "authors, datasets, or results. If a detail is unavailable, write 'Not available'.\n\n"
        f"PAPERS:\n{papers_summary}\n\n"
        "Write a well-structured literature review (3-5 paragraphs) synthesizing these papers, "
        "their relationships, and gaps."
    )
    review = generate_text(prompt_text, max_tokens=1200)

    table_prompt = (
        "Based ONLY on the papers below, produce a JSON array. Each object has keys: "
        "paper (title), method, dataset, result, limitation. Use 'Not available' if unknown. "
        "No commentary, no markdown.\n\n" + papers_summary
    )
    raw_table = generate_text(table_prompt, max_tokens=1000)
    raw_table_clean = re.sub(r"^```json|```$", "", raw_table.strip(), flags=re.MULTILINE).strip()
    try:
        table = json.loads(raw_table_clean)
    except:
        table = []
    
    return jsonify({"success": True, "literature_review": review, "lit_review_table": table})

@app.route("/api/generate_outline", methods=["POST"])
def api_generate_outline():
    data = request.json
    prompt = (
        f"Generate a numbered research paper outline for the title '{data.get('title') or 'Untitled'}' "
        f"in the domain {data.get('domain')}, research type {data.get('research_type')}. "
        "Use exactly these top-level sections in order: 1 Abstract, 2 Introduction, "
        "3 Literature Review, 4 Methodology, 5 Experiments, 6 Results, 7 Discussion, "
        "8 Conclusion, 9 Future Work, 10 References. Add 2-3 brief sub-points under each."
    )
    outline = generate_text(prompt, max_tokens=900)
    return jsonify({"success": True, "outline": outline})

@app.route("/api/generate_introduction", methods=["POST"])
def api_generate_introduction():
    data = request.json
    papers_summary = "\n".join(f"- {p.get('title')}: {p.get('abstract')}" for p in data.get("papers", [])) or "No papers uploaded."
    prompt = (
        f"Write an academic Introduction section for the paper titled '{data.get('title') or 'Untitled'}'.\n"
        f"Keywords: {', '.join(data.get('keywords', []))}\nResearch Gap: {data.get('gap', '')}\n"
        f"Reference paper summaries:\n{papers_summary}\n\n"
        "Cover: Background, Problem Statement, Research Gap, Objectives, Contributions, "
        "and Paper Organization, in flowing academic prose with clear paragraph breaks. "
        "Do not invent citations or facts not given above."
    )
    intro = generate_text(prompt, max_tokens=1200)
    return jsonify({"success": True, "introduction": intro})

@app.route("/api/generate_methodology", methods=["POST"])
def api_generate_methodology():
    data = request.json
    prompt = (
        "Write an academic Methodology section using strictly the details below. "
        "Do not invent datasets, models, or metrics beyond what is given. "
        "Explain the rationale, pipeline, and evaluation approach clearly.\n\n"
        f"Research Design: {data.get('m_design')}\nDataset: {data.get('m_dataset')}\n"
        f"Models: {data.get('m_models')}\nEvaluation Metrics: {data.get('m_metrics')}"
    )
    methodology = generate_text(prompt, max_tokens=1000)
    return jsonify({"success": True, "methodology": methodology})

@app.route("/api/generate_experiments", methods=["POST"])
def api_generate_experiments():
    data = request.json
    prompt = (
        "Write an academic Experiments section describing the setup below. "
        "Do not invent performance numbers; describe only the setup process.\n\n"
        f"Dataset: {data.get('e_dataset')}\nModels: {data.get('e_models')}\nEpochs: {data.get('e_epochs')}\n"
        f"Optimizer: {data.get('e_optimizer')}\nDetails: {data.get('e_details')}"
    )
    experiments = generate_text(prompt, max_tokens=900)
    return jsonify({"success": True, "experiments": experiments})

@app.route("/api/generate_chart", methods=["POST"])
def api_generate_chart():
    data = request.json
    exp_table = data.get("exp_table", [])
    chart_type = data.get("chart_type", "Bar")
    metric = data.get("metric", "Accuracy")

    if not exp_table:
        return jsonify({"success": False, "error": "No experiment data"})

    try:
        df = pd.DataFrame(exp_table)
        fig, ax = plt.subplots(figsize=(8, 5))

        if chart_type == "Bar":
            ax.bar(df["Model"], pd.to_numeric(df[metric], errors='coerce'), color="#4f6df5")
        elif chart_type == "Line":
            ax.plot(df["Model"], pd.to_numeric(df[metric], errors='coerce'), marker="o", color="#4f6df5", linewidth=2)
        elif chart_type == "Pie":
            ax.pie(pd.to_numeric(df[metric], errors='coerce'), labels=df["Model"], autopct="%1.1f%%", colors=["#4f6df5", "#7c5cff", "#1fae6e", "#f5a623"])

        ax.set_title(f"{metric} by Model", fontsize=14, pad=15)
        if chart_type != "Pie":
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig) # Prevent memory leaks in serverless container
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({"success": True, "image": f"data:image/png;base64,{encoded}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/generate_results", methods=["POST"])
def api_generate_results():
    data = request.json
    table_str = pd.DataFrame(data.get("exp_table", [])).to_string(index=False)
    prompt = (
        "Write a Results and Discussion section using ONLY the table data below. "
        "Do not invent numbers not present in the table.\n\n"
        f"{table_str}\n\n"
        "Cover: Observations, Strengths, Weaknesses, Future Improvements."
    )
    results = generate_text(prompt, max_tokens=1000)
    return jsonify({"success": True, "results": results})

@app.route("/api/generate_conclusion", methods=["POST"])
def api_generate_conclusion():
    data = request.json
    prompt = (
        f"Write a Conclusion section for the paper '{data.get('title') or 'Untitled'}' based on:\n"
        f"Introduction: {data.get('introduction', '')[:800]}\n"
        f"Methodology: {data.get('methodology', '')[:800]}\n"
        f"Results: {data.get('results', '')[:800]}\n\n"
        "Cover: Summary, Contributions, Limitations, Future Work."
    )
    conclusion = generate_text(prompt, max_tokens=800)
    return jsonify({"success": True, "conclusion": conclusion})

@app.route("/api/generate_abstract", methods=["POST"])
def api_generate_abstract():
    data = request.json
    prompt = (
        "Write a concise academic abstract (150-250 words) summarizing the following paper sections. "
        "Do not introduce new facts.\n\n"
        f"Introduction: {data.get('introduction', '')[:600]}\n"
        f"Literature Review: {data.get('literature_review', '')[:600]}\n"
        f"Methodology: {data.get('methodology', '')[:600]}\n"
        f"Results: {data.get('results', '')[:600]}\n"
        f"Conclusion: {data.get('conclusion', '')[:600]}"
    )
    abstract = generate_text(prompt, max_tokens=400)
    return jsonify({"success": True, "abstract": abstract})

@app.route("/api/export_zip", methods=["POST"])
def api_export_zip():
    P = request.json
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("main.tex", build_latex(P, P.get("journal", "IEEE")))
            zf.writestr("references.bib", build_bibtex(P.get("papers", [])) or "% No references added")
            zf.writestr("sections/abstract.tex", "\\begin{abstract}\n" + escape_latex(P.get("abstract", "Not available")) + "\n\\end{abstract}")
            zf.writestr("sections/introduction.tex", "\\section{Introduction}\n" + escape_latex(P.get("introduction", "Not available")))
            zf.writestr("sections/literature_review.tex", "\\section{Literature Review}\n" + escape_latex(P.get("literature_review", "Not available")))
            zf.writestr("sections/methodology.tex", "\\section{Methodology}\n" + escape_latex(P.get("methodology", "Not available")))
            zf.writestr("sections/results.tex", "\\section{Results and Discussion}\n" + escape_latex(P.get("results", "Not available")))
            zf.writestr("sections/conclusion.tex", "\\section{Conclusion}\n" + escape_latex(P.get("conclusion", "Not available")))
            zf.writestr("images/.gitkeep", "")
        buf.seek(0)
        filename = f"{safe_filename(P.get('title') or 'research_project')}.zip"
        return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/zip")
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)