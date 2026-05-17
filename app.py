import os
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from main import analyze_video as run_analysis
except Exception:
    run_analysis = None

app = Flask(__name__)
CORS(app)  # allow requests from React dev-server on :3000

# ── Folders ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(BASE_DIR, "input_videos")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_videos")
os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── POST /upload ──────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(INPUT_DIR, filename)
    name, original_ext = os.path.splitext(filename)

    # Analyzed output: use MP4 (OpenCV mp4v) — WebM/VP8 is often broken on Windows OpenCV builds.
    output_ext = ".mp4" if run_analysis is not None else original_ext
    output_filename = f"analyzed_{name}{output_ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Save uploaded file
    file.save(input_path)

    # Run analysis
    try:
        if run_analysis is None:
            shutil.copy(input_path, output_path)
        else:
            run_analysis(input_path, output_path)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 500

    return jsonify({
        "output_filename": output_filename,
        "input_path": input_path,
        "output_path": output_path,
    }), 200


# ── GET /video/<filename> ─────────────────────────────────────────────────────
_MIME_BY_EXT = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


@app.route("/video/<filename>")
def serve_video(filename):
    safe = secure_filename(os.path.basename(filename))
    if not safe:
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        return jsonify({"error": "Video not found"}), 404
    extension = os.path.splitext(safe)[1].lower()
    mimetype = _MIME_BY_EXT.get(extension)
    # conditional=True enables Range requests — required for reliable MP4 playback in browsers
    return send_file(path, mimetype=mimetype, conditional=True, max_age=0)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
