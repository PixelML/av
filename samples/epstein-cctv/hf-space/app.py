"""
Epstein Files CCTV — Video Memory Search
Search 200+ AI-generated temporal event captions from DOJ Epstein Files Dataset 8
(MCC prison CCTV surveillance footage) using natural language.
Click any result to play the exact moment.

Built with av (https://github.com/PixelML/av) — video memory for AI agents.
"""

import html
import re
import sqlite3

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

VIDEO_BASE = (
    "https://huggingface.co/datasets/"
    "PixelML/epstein-files-cctv-video-memory/resolve/main/videos"
)

# ── Startup ─────────────────────────────────────────────────────────────────

print("Downloading av.db ...", flush=True)
DB_PATH = hf_hub_download(
    repo_id="PixelML/epstein-files-cctv-video-memory",
    filename="av.db",
    repo_type="dataset",
)

# Load all artifact texts + metadata
print("Loading artifacts ...", flush=True)
_conn = sqlite3.connect(DB_PATH)
_cur = _conn.cursor()
_cur.execute(
    """
    SELECT a.text, a.start_sec, a.end_sec, a.type, v.filename
    FROM artifacts a
    JOIN videos v ON v.id = a.video_id
    ORDER BY v.filename, a.start_sec
    """
)
ARTIFACTS = [
    {"text": t, "start_sec": s or 0, "end_sec": e, "type": tp, "file": f}
    for t, s, e, tp, f in _cur.fetchall()
]
_conn.close()
print(f"  {len(ARTIFACTS)} artifacts loaded", flush=True)

# Embed everything with a local model — no API key needed
print("Loading embedding model ...", flush=True)
_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding {len(ARTIFACTS)} captions ...", flush=True)
INDEX_VECS = _model.encode(
    [a["text"] for a in ARTIFACTS],
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=128,
)
print(f"Index ready: {INDEX_VECS.shape}", flush=True)

# Warmup FTS5
_c = sqlite3.connect(DB_PATH)
_c.execute("SELECT count(*) FROM artifacts_fts")
_c.close()
print("Ready to serve!", flush=True)


# ── Search ──────────────────────────────────────────────────────────────────


def _fmt_ts(sec: float) -> str:
    return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"


def _fmt_range(start: float, end: float | None) -> str:
    s = _fmt_ts(start)
    if end and end != start:
        return f"{s}\u2013{_fmt_ts(end)}"
    return s


def semantic_search(query: str, limit: int = 20) -> list[dict]:
    """Encode query locally, cosine-rank against all artifact embeddings."""
    q_vec = _model.encode(query, normalize_embeddings=True)
    sims = INDEX_VECS @ q_vec
    top = np.argsort(sims)[::-1][:limit]
    results = []
    for rank, idx in enumerate(top, 1):
        idx = int(idx)
        a = ARTIFACTS[idx]
        results.append(
            {
                "rank": rank,
                "start_sec": a["start_sec"],
                "end_sec": a["end_sec"],
                "ts": _fmt_range(a["start_sec"], a["end_sec"]),
                "type": a["type"],
                "file": a["file"],
                "text": a["text"],
                "score": float(sims[idx]),
            }
        )
    return results


def fts_search(query: str, limit: int = 20) -> list[dict]:
    """SQLite FTS5 full-text search (fallback)."""
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return []
    fts_q = " OR ".join(f'"{t}"' for t in tokens)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT a.text, a.start_sec, a.end_sec, a.type, v.filename, rank
            FROM artifacts_fts fts
            JOIN artifacts a ON a.rowid = fts.rowid
            JOIN videos v ON v.id = a.video_id
            WHERE artifacts_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (fts_q, limit),
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()
    return [
        {
            "rank": i + 1,
            "start_sec": row["start_sec"] or 0,
            "end_sec": row["end_sec"],
            "ts": _fmt_range(row["start_sec"] or 0, row["end_sec"]),
            "type": row["type"],
            "file": row["filename"],
            "text": row["text"],
        }
        for i, row in enumerate(rows)
    ]


# ── Render ──────────────────────────────────────────────────────────────────

TYPE_BADGES = {
    "caption": ("\U0001f3ac", "#4a9eff"),
    "summary": ("\U0001f4cb", "#10b981"),
    "report": ("\U0001f4ca", "#f59e0b"),
    "transcript": ("\U0001f399", "#8b5cf6"),
}


def render(query: str, results: list[dict]) -> str:
    first = results[0]
    url0 = f"{VIDEO_BASE}/{first['file']}"
    q = html.escape(query)

    parts = [
        '<div class="av-wrap">'
        f'<video class="av-vid" controls autoplay muted preload="auto"'
        f' src="{url0}#t={int(first["start_sec"])}"></video>'
        f'<div class="av-info">{html.escape(first["file"])} \u2014 {first["ts"]}</div>'
        "</div>",
        f'<div style="font-size:13px;color:#888;margin-bottom:10px">'
        f'{len(results)} result{"s" if len(results) != 1 else ""}'
        f" for \u201c{q}\u201d</div>",
    ]

    for r in results:
        url = f"{VIDEO_BASE}/{r['file']}"
        cls = "av-r av-on" if r["rank"] == 1 else "av-r"
        score_html = ""
        if "score" in r:
            pct = int(r["score"] * 100)
            score_html = (
                f'<span style="font-size:11px;color:#666;margin-left:auto">{pct}%</span>'
            )
        emoji, color = TYPE_BADGES.get(r["type"], ("\u2022", "#888"))
        type_badge = (
            f'<span style="font-size:11px;background:{color}22;color:{color};'
            f'padding:1px 6px;border-radius:4px;font-weight:600">'
            f'{emoji} {r["type"]}</span>'
        )
        text_display = html.escape(r["text"])
        if r["type"] in ("summary", "report"):
            # Show first 300 chars with expand hint
            if len(r["text"]) > 300:
                text_display = html.escape(r["text"][:300]) + "\u2026"
        parts.append(
            f'<div class="{cls}" data-url="{url}" data-t="{r["start_sec"]}"'
            f' data-file="{html.escape(r["file"])}" data-ts="{r["ts"]}">'
            f'<div style="display:flex;gap:10px;align-items:baseline">'
            f'<span style="font-weight:700;color:#888;min-width:28px">#{r["rank"]}</span>'
            f"<span style=\"font-family:'Space Mono',monospace;font-size:13px;"
            f'color:#4a9eff;font-weight:600">{r["ts"]}</span>'
            f"{type_badge}{score_html}</div>"
            f'<div style="font-size:12px;color:#666;margin-top:2px">'
            f"{html.escape(r['file'])}</div>"
            f'<div style="font-size:13px;line-height:1.5;margin-top:6px">'
            f"{text_display}</div></div>"
        )
    return "\n".join(parts)


def search(query: str, limit: int = 10) -> str:
    if not query.strip():
        return EMPTY
    results = semantic_search(query, limit=limit)
    if not results:
        results = fts_search(query, limit=limit)
    if not results:
        q = html.escape(query)
        return (
            '<div style="text-align:center;padding:40px;color:#888">'
            f'No results for "<b>{q}</b>"<br>'
            '<span style="font-size:13px">'
            "Try: someone being escorted, tactical gear, person in red shirt, room goes empty"
            "</span></div>"
        )
    return render(query, results)


# ── UI ──────────────────────────────────────────────────────────────────────

HEAD = """
<script>
document.addEventListener('click', function(e) {
    var el = e.target.closest('.av-r');
    if (!el) return;
    var vid = document.querySelector('.av-vid');
    if (!vid) return;
    var url = el.dataset.url;
    var t = parseFloat(el.dataset.t);
    document.querySelectorAll('.av-r').forEach(function(r){r.classList.remove('av-on')});
    el.classList.add('av-on');
    var info = document.querySelector('.av-info');
    if (info) info.textContent = el.dataset.file + ' \\u2014 ' + el.dataset.ts;
    var cur = vid.currentSrc ? vid.currentSrc.split('/').pop().split('#')[0] : '';
    var next = url.split('/').pop();
    if (cur !== next) {
        vid.src = url;
        vid.onloadedmetadata = function(){ vid.currentTime = t; vid.play().catch(function(){}); };
        vid.load();
    } else {
        vid.currentTime = t;
        vid.play().catch(function(){});
    }
});
</script>
<style>
.av-r{padding:12px 16px;margin-bottom:6px;border:1px solid rgba(128,128,128,.2);border-radius:8px;cursor:pointer;transition:border-color .12s,background .12s}
.av-r:hover{border-color:rgba(74,158,255,.5);background:rgba(74,158,255,.04)}
.av-on{border-color:rgba(74,158,255,.7)!important;background:rgba(74,158,255,.08)!important}
.av-wrap{background:#000;border-radius:10px;overflow:hidden;margin-bottom:12px}
.av-wrap video{width:100%;max-height:420px;display:block}
.av-info{padding:8px 16px;font-size:12px;color:rgba(200,200,200,.6);background:rgba(0,0,0,.4);font-family:'Space Mono',monospace}
</style>
"""

EMPTY = (
    '<div style="text-align:center;padding:60px 20px;color:#555">'
    '<div style="font-size:48px;margin-bottom:16px">\u25b6</div>'
    '<div style="font-size:15px">Search to explore CCTV footage</div>'
    '<div style="font-size:13px;margin-top:8px">'
    "Try: someone being escorted out, tactical gear enters then room goes empty, person in red shirt"
    "</div></div>"
)

EXAMPLES = [
    ["someone being escorted out of the room"],
    ["tactical gear enters then the room goes empty"],
    ["who is the person in the red shirt and where do they go"],
    ["armed individual approaching a door"],
    ["room goes from crowded to completely empty"],
    ["officer using a radio or phone at the desk"],
    ["person carrying a bag enters the area"],
    ["multiple people rush in from the back"],
]

with gr.Blocks(
    title="Epstein Files CCTV \u2014 Video Memory Search",
    head=HEAD,
    theme=gr.themes.Base(
        primary_hue="blue",
        font=gr.themes.GoogleFont("Space Mono"),
    ),
) as demo:
    gr.Markdown(
        "# Epstein Files CCTV \u2014 Video Memory Search\n\n"
        "Search **209 AI-generated temporal event captions** + **7 structured reports** from DOJ Epstein Files "
        "Dataset 8 \u2014 MCC prison CCTV surveillance footage (July\u2013August 2019). "
        "**Click any result to play the exact moment.**\n\n"
        "Captions describe *what happens across frames* (temporal changes), not static scenes. "
        "Built with [av](https://github.com/PixelML/av) \u2014 video memory for AI agents. "
        "[agentic.video](https://agentic.video)\n\n---"
    )

    with gr.Row():
        query = gr.Textbox(
            label="Search query",
            placeholder="Try: someone being escorted out, tactical gear, person in red shirt...",
            scale=4,
        )
        limit = gr.Slider(1, 50, value=10, step=1, label="Max results", scale=1)

    btn = gr.Button("Search", variant="primary")
    output = gr.HTML(value=EMPTY)

    btn.click(fn=search, inputs=[query, limit], outputs=output)
    query.submit(fn=search, inputs=[query, limit], outputs=output)

    gr.Examples(examples=EXAMPLES, inputs=query, label="Try these searches")

    gr.Markdown(
        "\n---\n\n"
        "**Data source**: [DOJ Epstein Files \u2014 Data Set 8]"
        "(https://www.justice.gov/epstein/doj-disclosures/data-set-8-files) "
        "(419 CCTV surveillance videos from MCC New York). "
        "This demo indexes 10 clips (~4 hrs). "
        "Full dataset: [PixelML/epstein-files-cctv-video-memory]"
        "(https://huggingface.co/datasets/PixelML/epstein-files-cctv-video-memory)\n\n"
        "**How it works**: `pip install pixelml-av` \u2192 "
        "`av ingest video.mp4 --captions --topic security` \u2192 "
        '`av search "your query"`\n\n'
        "**Enterprise video intelligence**: "
        "[hello@pixelml.com](mailto:hello@pixelml.com)"
    )

if __name__ == "__main__":
    demo.launch()
