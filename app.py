from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    image_path = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )

    image.save(image_path)

    results = model(image_path)
    detections = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            fruit_name = model.names[cls]

            confidence = round(float(box.conf[0]) * 100, 2)

            detections.append({
                "name": fruit_name,
                "confidence": confidence
            })


    result_path = os.path.join(
        RESULT_FOLDER,
        image.filename
    )

    results[0].save(filename=result_path)

    return render_template(
    "index.html",
    result_image=image.filename,
    detections=detections
    )

if __name__ == "__main__":
    app.run(debug=True)