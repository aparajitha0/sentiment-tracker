"""
demo2.py - InsightFeed web dashboard with Landing Page + Analytics Workspace.
"""
import io
import tempfile

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from clean_data import load_and_clean
from sentiment import analyze, build_text, count_sentence, plot_clouds, plot_pie, sentiment_counts
from summarizer import add_summaries

ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"
RO_LABEL, VA_LABEL = "RoBERTa (accurate)", "VADER (fast)"
_clf = None


# ------------------------------------------------------------------ model
def get_roberta():
    """Loads the model once and reuses it."""
    global _clf
    if _clf is None:
        import torch
        from transformers import pipeline
        _clf = pipeline("sentiment-analysis", model=ROBERTA,
                        device=0 if torch.cuda.is_available() else -1,
                        truncation=True, max_length=512)
    return _clf


def label_roberta(df: pd.DataFrame, progress=None, chunk: int = 128) -> pd.DataFrame:
    """Adds sentiment + confidence using RoBERTa. Empty posts become Neutral."""
    out = df.copy()
    out["sentiment"], out["confidence"] = "Neutral", 0.0
    idx = out.index[out["text"].str.strip() != ""].tolist()
    clf = get_roberta()
    for start in range(0, len(idx), chunk):
        part = idx[start:start + chunk]
        preds = clf(out.loc[part, "text"].tolist(), batch_size=32)
        out.loc[part, "sentiment"] = [p["label"].capitalize() for p in preds]
        out.loc[part, "confidence"] = [round(p["score"], 3) for p in preds]
        if progress:
            progress(min(start + chunk, len(idx)) / max(len(idx), 1), desc="Labelling posts…")
    return out


# ------------------------------------------------------------------ helpers
def _to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _kpi_html(df: pd.DataFrame, model_choice: str, note: str) -> str:
    c = sentiment_counts(df)
    n = max(len(df), 1)
    pos, neg, neu = int(c["Positive"]), int(c["Negative"]), int(c["Neutral"])
    return f"""
    <div style="margin-bottom: 14px;">
      <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px;">
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-size:12px; font-weight:600; color:#64748b; text-transform:uppercase;">Total Posts</div>
          <div style="font-size:24px; font-weight:800; color:#4f46e5; margin-top:4px;">{len(df):,}</div>
        </div>
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-size:12px; font-weight:600; color:#15803d; text-transform:uppercase;">Positive</div>
          <div style="font-size:24px; font-weight:800; color:#16a34a; margin-top:4px;">{pos:,} <span style="font-size:14px;">({pos/n:.1%})</span></div>
        </div>
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-size:12px; font-weight:600; color:#b91c1c; text-transform:uppercase;">Negative</div>
          <div style="font-size:24px; font-weight:800; color:#dc2626; margin-top:4px;">{neg:,} <span style="font-size:14px;">({neg/n:.1%})</span></div>
        </div>
        <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-size:12px; font-weight:600; color:#475569; text-transform:uppercase;">Neutral</div>
          <div style="font-size:24px; font-weight:800; color:#334155; margin-top:4px;">{neu:,} <span style="font-size:14px;">({neu/n:.1%})</span></div>
        </div>
      </div>
      <p style="margin-top:10px; font-size:14px; color:#475569;">
        <b>Summary:</b> {count_sentence(df)}{note} &nbsp;|&nbsp; <b>Engine:</b> {model_choice}
      </p>
    </div>
    """


# ------------------------------------------------------------------ pipeline
def run(file, model_choice, max_posts, common_word_filter, api_key, progress=None):
    """Upload -> clean -> label -> summarise -> KPI cards, pie, clouds, CSV."""
    prog = progress or (lambda *a, **k: None)
    if file is None:
        raise gr.Error("Please upload a dataset first.")
    path = file if isinstance(file, str) else file.name

    prog(0.02, desc="Cleaning data…")
    try:
        df = load_and_clean(path, verbose=False)
    except Exception as e:
        raise gr.Error(f"Could not read this file: {e}")
    if df.empty:
        raise gr.Error("No text found in this file.")

    note = ""
    if len(df) > max_posts:
        df = df.sample(int(max_posts), random_state=1).sort_index().reset_index(drop=True)
        note = f" (random sample of {int(max_posts):,} posts)"

    if model_choice == RO_LABEL:
        df["text"] = build_text(df)
        df = label_roberta(df, progress=lambda f, desc="": prog(0.05 + 0.75 * f, desc=desc))
    else:
        prog(0.3, desc="Labelling posts…")
        df = analyze(df, neutral_zone=0.05)

    prog(0.85, desc="Writing summaries…")
    df = add_summaries(df, api_key=api_key, ai_limit=50)

    prog(0.93, desc="Drawing charts…")
    kpi_cards = _kpi_html(df, model_choice, note)
    pie = plot_pie(df)
    figs = plot_clouds(df, max_doc_freq=common_word_filter, show=False)
    clouds = [(_to_image(f), name) for name, f in figs.items() if f is not None]

    result = df[["post_id", "sentiment", "summary"]]
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", prefix="insightfeed_results_", delete=False)
    result.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    return kpi_cards, pie, clouds, result.head(200), tmp.name


# ------------------------------------------------------------------ page navigation
def go_to_workspace():
    return gr.update(visible=False), gr.update(visible=True)


def go_to_home():
    return gr.update(visible=True), gr.update(visible=False)


# ------------------------------------------------------------------ web page
def launch(share: bool = True, preload: bool = True):
    if preload:
        get_roberta()

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate")

    with gr.Blocks(theme=theme, title="InsightFeed — Sentiment Intelligence") as app:

        # ==================== PAGE 1: LANDING PAGE ====================
        with gr.Column(visible=True) as landing_page:
            gr.HTML("""
            <div style="max-width: 960px; margin: 24px auto 12px auto; text-align: center;">
              <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                          padding: 52px 36px; border-radius: 20px; color: white;
                          box-shadow: 0 12px 30px rgba(79, 70, 229, 0.22);">
                <div style="display:inline-block; background: rgba(255,255,255,0.16);
                            padding: 6px 14px; border-radius: 999px; font-size: 13px;
                            font-weight: 600; letter-spacing: 0.4px; margin-bottom: 14px;">
                  AI-POWERED SENTIMENT TRACKER
                </div>
                <h1 style="margin: 0; font-size: 46px; font-weight: 900; letter-spacing: -1px;">
                  📊 InsightFeed
                </h1>
                <p style="margin: 14px auto 0 auto; max-width: 660px; font-size: 18px; line-height: 1.6; opacity: 0.94;">
                  Turn raw, unstructured social media posts into clear public sentiment metrics,
                  theme clouds, and executive post summaries in seconds.
                </p>
              </div>

              <h2 style="margin: 34px 0 16px 0; font-size: 22px; font-weight: 700; color: #1e293b;">
                Platform Objectives
              </h2>

              <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; text-align: left;">
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                  <div style="font-size: 22px; margin-bottom: 6px;">🧹</div>
                  <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Schema-Agnostic Ingestion</div>
                  <p style="margin: 6px 0 0 0; font-size: 14px; color: #475569; line-height: 1.5;">
                    Upload any CSV, Excel, or JSON dataset—automatically detects text columns, strips noise, and preserves emojis and emphasis.
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                  <div style="font-size: 22px; margin-bottom: 6px;">🧠</div>
                  <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Context-Aware Sentiment AI</div>
                  <p style="margin: 6px 0 0 0; font-size: 14px; color: #475569; line-height: 1.5;">
                    Powered by fine-tuned RoBERTa transformers (with a fast VADER baseline) to classify Positive, Negative, and Neutral sentiment accurately.
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                  <div style="font-size: 22px; margin-bottom: 6px;">☁️</div>
                  <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Dynamic Topic & Keyword Discovery</div>
                  <p style="margin: 6px 0 0 0; font-size: 14px; color: #475569; line-height: 1.5;">
                    Filters out dataset-wide filler terms automatically using document-frequency thresholds to surface real drivers behind each sentiment.
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                  <div style="font-size: 22px; margin-bottom: 6px;">📑</div>
                  <div style="font-size: 16px; font-weight: 700; color: #0f172a;">Instant Summaries & Export</div>
                  <p style="margin: 6px 0 0 0; font-size: 14px; color: #475569; line-height: 1.5;">
                    Generates concise post-level summaries and lets teams download clean, ready-to-share CSV reports with a single click.
                  </p>
                </div>
              </div>
            </div>
            """)

            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    start_btn = gr.Button("Go to Analyse →", variant="primary", size="lg")
                gr.Column(scale=1)

        # ==================== PAGE 2: ANALYTICS WORKSPACE ====================
        with gr.Column(visible=False) as workspace_page:
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                                padding: 18px 24px; border-radius: 14px; color: white; margin-bottom: 12px;">
                      <h1 style="margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px;">
                        📊 InsightFeed — Analytics Workspace
                      </h1>
                      <p style="margin: 4px 0 0 0; font-size: 14px; opacity: 0.92;">
                        Upload any social media dataset (CSV, Excel, JSON) to classify public sentiment, extract key themes, and export summarized insights.
                      </p>
                    </div>
                    """)
                with gr.Column(scale=1, min_width=140):
                    back_btn = gr.Button("← Home", variant="secondary", size="sm")

            with gr.Row():
                # Left Sidebar: Controls
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 1. Upload & Configure")
                    file = gr.File(label="Dataset (CSV / Excel / JSON)", type="filepath")
                    model = gr.Radio([RO_LABEL, VA_LABEL], value=RO_LABEL, label="Sentiment Engine")
                    max_posts = gr.Slider(50, 6000, value=1000, step=50,
                                          label="Max Posts to Analyse")
                    common = gr.Slider(0.05, 0.6, value=0.15, step=0.05,
                                       label="Auto-Stopword Filter (Max Doc Freq)")
                    with gr.Accordion("⚙️ Optional: AI Abstractive Summaries", open=False):
                        key = gr.Textbox(type="password", label="Anthropic API Key",
                                         placeholder="sk-ant-... (first 50 posts)")
                    btn = gr.Button("🚀 Analyze Sentiment", variant="primary", size="lg")

                # Right Canvas: Dashboard Outputs
                with gr.Column(scale=3):
                    kpi_out = gr.HTML(
                        "<div style='padding:36px; text-align:center; border:2px dashed #cbd5e1; border-radius:12px; color:#64748b;'>"
                        "Upload a dataset on the left and click <b>Analyze Sentiment</b> to view insights.</div>"
                    )
                    with gr.Tabs():
                        with gr.TabItem("📈 Visual Analytics"):
                            with gr.Row():
                                pie = gr.Plot(label="Sentiment Distribution")
                                clouds = gr.Gallery(label="Class-Wise Word Clouds", columns=2, height=400)
                        with gr.TabItem("📋 Labeled Posts & Export"):
                            dl = gr.File(label="Download Full Labeled CSV")
                            table = gr.Dataframe(label="Preview (First 200 Rows)", wrap=True)

            btn.click(run, [file, model, max_posts, common, key], [kpi_out, pie, clouds, table, dl])

        # Navigation events
        start_btn.click(go_to_workspace, inputs=None, outputs=[landing_page, workspace_page])
        back_btn.click(go_to_home, inputs=None, outputs=[landing_page, workspace_page])

    app.launch(share=share, debug=False)
