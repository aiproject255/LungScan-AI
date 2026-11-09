import os
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

MODEL_PATH = "Lung_Cancer_AI_Model.keras"

# Load your actual trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_nodule(image):
    if image is None:
        return "Please upload a CT scan image."

    # Match the preprocessing used during training
    image = image.convert("L")
    image = image.resize((128, 128))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(1, 128, 128, 1)

    # Real prediction from your trained CNN
    prediction = model.predict(image_array, verbose=0)
    score = float(prediction[0][0])

    if score >= 0.5:
        result = "Nodule Detected"
    else:
        result = "No Nodule Detected"

    return f"{result}\nModel output: {score:.4f}"

with gr.Blocks(title="LungScan AI") as demo:

    gr.Markdown("""
    # 🫁 LungScan AI
    ### Lung CT Nodule Detection Using Deep Learning

    Upload a lung CT scan image to analyze it using
    our trained Convolutional Neural Network (CNN).

    **Class 12 CBSE Artificial Intelligence Project**
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Lung CT Image",
                type="pil"
            )

            analyze_button = gr.Button(
                "🔍 Analyze Image",
                variant="primary"
            )

        with gr.Column():
            result_output = gr.Textbox(
                label="AI Analysis Result",
                lines=3
            )

    analyze_button.click(
        fn=predict_nodule,
        inputs=image_input,
        outputs=result_output
    )

    gr.Markdown("""
    ---
    ### About This Project

    **AI Model:** Convolutional Neural Network (CNN)

    **Input:** 128 × 128 grayscale CT image

    **Test Accuracy:** 89.74% on 39 held-out images

    **Framework:** TensorFlow / Keras

    **Purpose:** Educational demonstration of
    lung CT slice nodule classification.

    ⚠️ **Disclaimer:** This is a school project,
    not a medical diagnostic tool. A nodule prediction
    does not establish whether cancer is present.
    """)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
