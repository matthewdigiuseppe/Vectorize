#!/usr/bin/env python3
"""
app.py — Web dashboard for the Vectorize image-to-SVG tool.

Run with:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import io
import os
import uuid
import time
import threading
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from PIL import Image
from werkzeug.utils import secure_filename

from vectorize import SUPPORTED_EXTENSIONS, vectorize_pil_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store: job_id -> {status, filename, svg_filename, error, params, ...}
jobs: dict[str, dict] = {}

# Cleanup files older than 30 minutes
CLEANUP_INTERVAL = 600  # seconds
MAX_FILE_AGE = 1800  # seconds


def _cleanup_old_files() -> None:
    """Periodically remove stale uploads and outputs."""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        now = time.time()
        for d in (UPLOAD_DIR, OUTPUT_DIR):
            for f in d.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > MAX_FILE_AGE:
                    f.unlink(missing_ok=True)
        # Purge old jobs from memory
        stale = [jid for jid, j in jobs.items() if now - j.get("created", now) > MAX_FILE_AGE]
        for jid in stale:
            jobs.pop(jid, None)


threading.Thread(target=_cleanup_old_files, daemon=True).start()


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/convert", methods=["POST"])
def api_convert():
    """Accept an image upload and conversion parameters, return SVG."""
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify(error="No file selected"), 400

    if not _allowed_file(file.filename):
        return jsonify(error=f"Unsupported format. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"), 400

    mode = request.form.get("mode", "outline")
    if mode not in ("outline", "color", "detailed"):
        return jsonify(error="Invalid mode"), 400

    try:
        n_colors = int(request.form.get("colors", "8"))
        simplify = float(request.form.get("simplify", "0.3"))
        threshold_raw = request.form.get("threshold", "")
        threshold = int(threshold_raw) if threshold_raw else None
    except ValueError:
        return jsonify(error="Invalid numeric parameter"), 400

    if not 0.0 <= simplify <= 1.0:
        return jsonify(error="simplify must be between 0 and 1"), 400
    if n_colors < 2:
        return jsonify(error="colors must be at least 2"), 400
    if threshold is not None and not 0 <= threshold <= 255:
        return jsonify(error="threshold must be between 0 and 255"), 400

    # Save uploaded file
    job_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file.save(str(upload_path))

    try:
        img = Image.open(upload_path)
        svg_content = vectorize_pil_image(img, mode, n_colors, simplify, threshold)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 500

    svg_filename = f"{job_id}_{Path(safe_name).stem}.svg"
    svg_path = OUTPUT_DIR / svg_filename
    svg_path.write_text(svg_content)

    # Build a thumbnail-sized data URI of the original for the history panel
    img.thumbnail((120, 120))
    thumb_buf = io.BytesIO()
    img.save(thumb_buf, format="PNG")
    import base64
    thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()

    original_size = upload_path.stat().st_size
    svg_size = svg_path.stat().st_size

    jobs[job_id] = {
        "status": "done",
        "filename": safe_name,
        "svg_filename": svg_filename,
        "mode": mode,
        "colors": n_colors,
        "simplify": simplify,
        "threshold": threshold,
        "original_size": original_size,
        "svg_size": svg_size,
        "created": time.time(),
    }

    return jsonify(
        job_id=job_id,
        svg_url=f"/outputs/{svg_filename}",
        svg_content=svg_content,
        filename=safe_name,
        mode=mode,
        original_size=original_size,
        svg_size=svg_size,
        thumb=f"data:image/png;base64,{thumb_b64}",
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/history")
def api_history():
    """Return recent conversion jobs."""
    recent = sorted(jobs.values(), key=lambda j: j.get("created", 0), reverse=True)[:20]
    return jsonify(history=recent)


if __name__ == "__main__":
    print("Vectorize Dashboard running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
