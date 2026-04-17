from flask import Flask, render_template, request
import os, cv2, numpy as np, sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DB = "database.db"

model = load_model("model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY,
                 filename TEXT,
                 result TEXT,
                 confidence INTEGER,
                 date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------------- FACE EXTRACTION ----------------
def extract_face(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    return face

# ---------------- PREDICTION ----------------
def detect(path):
    face = extract_face(path)

    if face is None:
        return "No Face Detected", 0

    face = face / 255.0
    face = face.reshape(1, 224, 224, 3)

    pred = model.predict(face)[0][0]

    print("Prediction value:", pred)

    # FIXED LABEL MAPPING
    if pred > 0.5:
        return "Real", int(pred * 100)
    else:
        return "Fake", int((1 - pred) * 100)

# ---------------- SAVE TO DB ----------------
def save_db(name, res, conf):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES(NULL,?,?,?,?)",
              (name, res, conf, datetime.now()))
    conn.commit()
    conn.close()

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    pred = None

    if request.method == "POST":
        f = request.files["image"]

        if f:
            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)

            res, conf = detect(path)
            save_db(f.filename, res, conf)

            pred = f"{res} ({conf}%)"

    return render_template("index.html", prediction=pred)


@app.route("/history")
def history():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    data = c.execute("SELECT * FROM results ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("history.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)