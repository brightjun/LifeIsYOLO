from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import requests
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

YOLO_MODEL = YOLO("yolov8n.pt")


def load_image_from_url(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def predict_objects_from_image(img):
    try:
        # Use the pre-loaded YOLO model for prediction
        results = YOLO_MODEL.predict(img)
        # Convert results to JSON format
        result_json = results[0].tojson()
        img_size = results[0].orig_shape
        # Create a dictionary with image size and labels
        result_dict = {"imgSize": img_size, "Labels": result_json}

        return result_dict

    except Exception as e:
        return {"error": str(e)}


@app.route("/predict/<path:image_url>", methods=["GET"])
def predict_with_parameter(image_url):
    try:
        img = load_image_from_url(image_url)
        result = predict_objects_from_image(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
