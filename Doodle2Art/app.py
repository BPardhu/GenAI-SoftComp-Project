# app.py
import gradio as gr
from analyzer import analyze_doodle
from generator import generate_art


def doodle_to_art(image):
    # Save user doodle
    image.save("input_doodle.png")

    # Preprocess doodle
    processed = analyze_doodle("input_doodle.png")

    # Universal watercolor prompt (no subject restriction)
    prompt = (
        "a watercolor painting of the drawn subject, "
        "clean outline, soft colors, artistic, centered composition"
    )

    # Generate artwork
    output = generate_art(processed, prompt)
    return output


ui = gr.Interface(
    fn=doodle_to_art,
    inputs=gr.Image(type="pil", label="Draw or Upload Doodle"),
    outputs=gr.Image(label="Generated Watercolor Artwork"),
    title="🎨 Doodle2Art – Watercolor Generator",
    description=(
        "Draw or upload any doodle (animal, human, object, or abstract) "
        "and let AI transform it into a watercolor painting."
    )
)

ui.launch()