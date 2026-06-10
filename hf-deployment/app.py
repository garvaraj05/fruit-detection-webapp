import gradio as gr
from ultralytics import YOLO
from PIL import Image
import tempfile

model = YOLO("best.pt")

def detect(image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)

        results = model(tmp.name)

        plotted = results[0].plot()

        return Image.fromarray(plotted)

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🍎 Fruit Detection using YOLOv8",
    description="Upload an image and detect fruits."
)

demo.launch()