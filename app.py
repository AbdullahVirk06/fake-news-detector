# app.py

# ---------------------------
# 1) Install dependencies (run once) - Not needed for deployment on Spaces
# ---------------------------

# ---------------------------
# 2) Imports & API keys
# ---------------------------
import os
import json
import requests
import gradio as gr
import tempfile
import sqlite3
import datetime
from pathlib import Path
import pandas as pd
import random
import plotly.express as px

# File parsing libs
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract

# Groq client
from groq import Groq

# NewsAPI client
from newsapi import NewsApiClient

# ---------------------------
# === CHANGES FOR DEPLOYMENT ===
# Use environment variables/secrets instead of hardcoded keys
# ---------------------------
SENTIMENT_KEY = os.environ.get("SENTIMENT_KEY")
FAKENEWS_KEY = os.environ.get("FAKENEWS_KEY")
GROQ_KEY = os.environ.get("GROQ_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

MONGO_URI = os.environ.get("MONGO_URI", "")

# ---------------------------
# 3) Model endpoints
# ---------------------------
SENTIMENT_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
FAKENEWS_URL = "https://api-inference.huggingface.co/models/Pulk17/Fake-News-Detection"

# ---------------------------
# 4) Database setup (MongoDB optional / SQLite fallback)
# ---------------------------
DB_SQLITE_FILE = "analyses.sqlite3"
if MONGO_URI:
    try:
        from pymongo import MongoClient
        mc = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mc.get_database()
        col = db.get_collection("analyses")
        mc.server_info()
        USING_MONGO = True
    except Exception as e:
        print("MongoDB connection failed; falling back to SQLite. Error:", e)
        USING_MONGO = False
else:
    USING_MONGO = False

# Create sqlite table if required
conn = sqlite3.connect(DB_SQLITE_FILE, check_same_thread=False)
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    text_snippet TEXT,
    sentiment_label TEXT,
    sentiment_conf REAL,
    fake_label TEXT,
    fake_conf REAL,
    reasoning TEXT,
    related_articles TEXT
)
''')
conn.commit()

def save_result_db(payload):
    """Try MongoDB first; if not available, use SQLite."""
    try:
        if USING_MONGO:
            col.insert_one(payload)
            return True
    except Exception as e:
        print("Mongo insert failed:", e)
    # fallback sqlite
    try:
        cur.execute(
            "INSERT INTO analyses (timestamp, text_snippet, sentiment_label, sentiment_conf, fake_label, fake_conf, reasoning, related_articles) VALUES (?,?,?,?,?,?,?,?)",
            (
                payload.get("timestamp"),
                payload.get("text")[:400],
                payload.get("sentiment_label"),
                payload.get("sentiment_confidence"),
                payload.get("fake_label"),
                payload.get("fake_confidence"),
                payload.get("reasoning"),
                json.dumps(payload.get("related_articles", []))
            )
        )
        conn.commit()
        return True
    except Exception as e:
        print("SQLite insert failed:", e)
        return False

def get_history_db():
    if USING_MONGO:
        records = list(col.find().sort("timestamp", -1).limit(20))
        df_data = []
        for rec in records:
            df_data.append({
                "Timestamp": rec.get("timestamp"),
                "Text Snippet": rec.get("text", "")[:75] + "...",
                "Sentiment": f"{rec.get('sentiment_label')} ({rec.get('sentiment_confidence', 0):.2f}%)",
                "Fake News": f"{rec.get('fake_label')} ({rec.get('fake_confidence', 0):.2f}%)",
                "Reasoning": rec.get("reasoning", "")[:75] + "..."
            })
        if df_data:
            return pd.DataFrame(df_data)
        return pd.DataFrame()
    else:
        cur.execute("SELECT timestamp, text_snippet, sentiment_label, sentiment_conf, fake_label, fake_conf, reasoning FROM analyses ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall()
        df_data = []
        for row in rows:
            df_data.append({
                "Timestamp": row[0],
                "Text Snippet": row[1],
                "Sentiment": f"{row[2]} ({row[3]:.2f}%)",
                "Fake News": f"{row[4]} ({row[5]:.2f}%)",
                "Reasoning": row[6]
            })
        if df_data:
            return pd.DataFrame(df_data)
        return pd.DataFrame()

# ---------------------------
# 5) Helpers: file text extractor
# ---------------------------
def extract_text_from_filepath(filepath):
    """Support .txt, .docx, .pdf, .png/.jpg (OCR)."""
    p = Path(filepath)
    suffix = p.suffix.lower()
    try:
        if suffix == ".txt":
            return Path(filepath).read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".docx":
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        elif suffix == ".pdf":
            reader = PyPDF2.PdfReader(filepath)
            texts = []
            for pg in reader.pages:
                try:
                    t = pg.extract_text()
                    if t:
                        texts.append(t)
                except:
                    pass
            return "\n".join(texts)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)
            return text
        else:
            return ""
    except Exception as e:
        return f"⚠️ Error extracting text: {e}"

# ---------------------------
# 6) Hugging Face API wrappers (robust)
# ---------------------------
def hf_post(url, api_key, text, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return {"error": f"Network error: {e}"}
    if r.status_code != 200:
        return {"error": f"HF API status {r.status_code}", "raw": r.text}
    try:
        return r.json()
    except Exception as e:
        return {"error": f"JSON parse error: {e}", "raw": r.text}

def call_sentiment(text):
    return hf_post(SENTIMENT_URL, SENTIMENT_KEY, text)

def call_fake_news(text):
    return hf_post(FAKENEWS_URL, FAKENEWS_KEY, text)

# ---------------------------
# 7) Groq reasoning wrapper
# ---------------------------
groq_client = Groq(api_key=GROQ_KEY)

def get_groq_reasoning(text, prediction):
    prompt = f"Text: {text}\nPrediction: {prediction}\nExplain in 2-4 clear sentences WHY this prediction makes sense (cite typical signals)."
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a concise fact-checking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=220
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"

# ---------------------------
# 8) NewsAPI helper (related articles)
# ---------------------------
newsapi_client = NewsApiClient(api_key=NEWSAPI_KEY)

def fetch_related_articles(query, max_results=4):
    if not newsapi_client:
        return []
    try:
        res = newsapi_client.get_everything(q=query, language="en", sort_by="relevancy", page_size=max_results)
        articles = res.get("articles", []) if isinstance(res, dict) else []
        simplified = [{"source": a.get("source",{}).get("name"), "title": a.get("title"), "url": a.get("url")} for a in articles]
        return simplified
    except Exception as e:
        return [{"error": str(e)}]

# ---------------------------
# 9) Pretty formatters
# ---------------------------
def pretty_sentiment(hf_resp):
    if isinstance(hf_resp, dict) and hf_resp.get("error"):
        return ("Error", 0, hf_resp.get("raw", hf_resp.get("error")))
    try:
        scores = hf_resp[0]
        best = max(scores, key=lambda x: x["score"])
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        label = label_map.get(best["label"], best["label"])
        conf = round(best["score"] * 100, 2)
        return (label, conf, None)
    except Exception as e:
        return ("ParseError", 0, str(e))

def pretty_fake(hf_resp):
    if isinstance(hf_resp, dict) and hf_resp.get("error"):
        return ("Error", 0, hf_resp.get("raw", hf_resp.get("error")))
    try:
        scores = hf_resp[0]
        best = max(scores, key=lambda x: x["score"])
        label_map = {"LABEL_0": "Real News", "LABEL_1": "Fake News", "0": "Fake News", "1": "Real News"}
        label = label_map.get(best["label"], best["label"])
        conf = round(best["score"] * 100, 2)
        return (label, conf, None)
    except Exception as e:
        return ("ParseError", 0, str(e))

# ---------------------------
# 10) Report saving & returning download
# ---------------------------
def build_report_text(text, sentiment_label, sentiment_conf, fake_label, fake_conf, reasoning, related_articles):
    lines = []
    lines.append("📰 News Analysis Report")
    lines.append("="*40)
    lines.append("")
    lines.append("Input text:")
    lines.append(text)
    lines.append("")
    lines.append(f"Sentiment: {sentiment_label} ({sentiment_conf}%)")
    lines.append(f"Fake news prediction: {fake_label} ({fake_conf}%)")
    lines.append("")
    lines.append("Reasoning:")
    lines.append(reasoning)
    lines.append("")
    if related_articles:
        lines.append("Related Articles:")
        for a in related_articles:
            if "error" in a:
                lines.append(f"- Error: {a['error']}")
            else:
                lines.append(f"- {a.get('title')} ({a.get('source')}) - {a.get('url')}")
    return "\n".join(lines)

def save_report_to_file(report_text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(report_text)
    tmp.flush()
    tmp.close()
    return tmp.name

# ---------------------------
# 11) Main pipeline (called by UI)
# ---------------------------
def analyze_pipeline(user_text, file_path):
    text = (user_text or "").strip()
    if file_path:
        if isinstance(file_path, (list, tuple)):
            file_path = file_path[0]
        extracted = extract_text_from_filepath(file_path)
        if extracted and extracted.strip():
            text = extracted

    if not text:
        gr.Warning("⚠️ Please provide text or upload a file to analyze.")
        return "", gr.File(visible=False, value=None), None, None, gr.update(visible=False), None, gr.update(selected=0)
        
    raw_sent = call_sentiment(text)
    sent_label, sent_conf, sent_err = pretty_sentiment(raw_sent)

    raw_fake = call_fake_news(text)
    fake_label, fake_conf, fake_err = pretty_fake(raw_fake)

    reasoning = get_groq_reasoning(text, fake_label) if not fake_err else f"Reasoning unavailable: {fake_err}"

    short_q = " ".join(text.split()[:8])
    related = fetch_related_articles(short_q)

    report_text = build_report_text(text, sent_label, sent_conf, fake_label, fake_conf, reasoning, related)
    report_filepath = save_report_to_file(report_text)

    payload = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "text": text,
        "sentiment_label": sent_label,
        "sentiment_confidence": sent_conf,
        "fake_label": fake_label,
        "fake_confidence": fake_conf,
        "reasoning": reasoning,
        "related_articles": related
    }
    saved = save_result_db(payload)
    db_status = f"**Saved to DB:** {'MongoDB' if USING_MONGO else 'SQLite'} — { '✅' if saved else '❌' }"

    md = f"""## 📰 Analysis Summary

**Sentiment:** {sent_label} ({sent_conf}%)
**Fake News Prediction:** {fake_label} ({fake_conf}%)

**Reasoning:**
{reasoning}

**Related Articles:**
"""
    if related:
        for a in related:
            if "error" in a:
                md += f"- {a['error']}\n"
            else:
                md += f"- [{a.get('title')}]({a.get('url')}) — *{a.get('source')}*\n"
    else:
        md += "- No related articles found.\n"
    md += f"\n{db_status}"

    sent_df = pd.DataFrame({'Label': [sent_label, 'Other'], 'Confidence': [sent_conf, 100 - sent_conf]})
    fig = px.pie(sent_df, values='Confidence', names='Label', title='Sentiment Confidence', color_discrete_sequence=['#4B0082', '#E0E0E0'])
    sent_plot = gr.Plot(fig)

    fake_df = pd.DataFrame({'Label': [fake_label, 'Other'], 'Confidence': [fake_conf, 100 - fake_conf]})
    fig_fake = px.pie(fake_df, values='Confidence', names='Label', title='Fake News Confidence', color_discrete_sequence=['#8B0000', '#E0E0E0'])
    fake_plot = gr.Plot(fig_fake)

    return (
        md,
        gr.File(value=report_filepath, visible=True),
        sent_plot,
        fake_plot,
        gr.update(visible=True),
        get_history_db(),
        gr.update(selected=1)
    )

def clear_all():
    return (
        "",
        None,
        "",
        gr.File(visible=False, value=None),
        gr.update(visible=False),
        None,
        None,
        gr.update(selected=0)
    )

def refresh_history():
    return get_history_db()

# ---------------------------
# 12) Gradio UI (enhanced design)
# ---------------------------
with gr.Blocks(theme=gr.themes.Monochrome(), title="News Analysis App", css="""
.center-heading {
    text-align: center;
    font-size: 2.5em;
    color: #7c5cff;
}
.result-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    background-color: #f9f9f9;
    margin-bottom: 20px;
}
.gr-button-primary {
    background-color: #4CAF50;
    border-color: #4CAF50;
}
.gr-button-secondary {
    background-color: #6c757d;
    border-color: #6c757d;
}
""") as demo:

    gr.Markdown("<h1 class='center-heading'>📰 Fake News & Sentiment Analyzer</h1>")
    gr.Markdown("Analyze news articles, headlines, or social media posts for **sentiment** and **credibility**. The app uses multiple AI models to provide a detailed report, reasoning, and related articles.")
    
    with gr.Tabs(elem_id="tabs") as tabs:
        with gr.TabItem("🚀 Analyze News"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(label="Paste text here", lines=10, placeholder="Paste article, headline, or social post...")
                    file_input = gr.File(label="Or upload a file (txt, docx, pdf, png, jpg)", type="filepath")
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    gr.Markdown(f"**Database:** {'MongoDB (connected)' if USING_MONGO else 'SQLite (local)'}")
                    with gr.Accordion("Quick Examples", open=False):
                        gr.Examples(
                            examples=[
                                ["Aliens have landed in Paris and are negotiating with world leaders."],
                                ["Local election results show candidate X won by a narrow margin amid claims of voter suppression."],
                                ["PM's Coordinator Rana Ihsan Afzal says agreement will open the door for other Arab countries to also join. Islamabad, Riyadh this week signed a pact pledging aggression against one will be treated as attack on both"]
                            ],
                            inputs=input_text
                        )

                with gr.Column(scale=2, visible=False) as result_display_col:
                    gr.Markdown("### Analysis Summary")
                    with gr.Row():
                        with gr.Column():
                            sent_chart = gr.Plot(label="Sentiment Breakdown")
                        with gr.Column():
                            fake_chart = gr.Plot(label="Fake News Confidence")
                    result_markdown = gr.Markdown("Results will appear here.")
                    download_file = gr.File(label="Download Full Report", visible=False)

        with gr.TabItem("📊 Analysis History"):
            gr.Markdown("### Recent Analyses")
            history_table = gr.Dataframe(
                headers=["Timestamp", "Text Snippet", "Sentiment", "Fake News", "Reasoning"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )
            with gr.Row():
                refresh_history_btn = gr.Button("Refresh History", variant="secondary")
                
    analyze_btn.click(
        analyze_pipeline,
        inputs=[input_text, file_input],
        outputs=[
            result_markdown,
            download_file,
            sent_chart,
            fake_chart,
            result_display_col,
            history_table,
            tabs
        ]
    )

    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[
            input_text, 
            file_input, 
            result_markdown,
            download_file,
            result_display_col,
            sent_chart,
            fake_chart,
            tabs
        ]
    )
    
    refresh_history_btn.click(
        refresh_history,
        outputs=history_table
    )
    
demo.launch()