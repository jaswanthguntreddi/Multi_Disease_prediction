from flask import Flask, render_template, request
import requests
import base64
import os
import markdown

app = Flask(__name__)

GROQ_API_KEY = "api-key"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.form["prompt"]
    image_file = request.files.get("image")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    content = [{"type": "text", "text": prompt}]
    if image_file:
        base64_image = encode_image(image_file)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 1,
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    try:
        output = response.json()
        answer = output["choices"][0]["message"]["content"]
        html_response = markdown.markdown(answer)  # Converts markdown to HTML
    except Exception as e:
        html_response = f"<strong>Error:</strong> {e}<br><br><code>{response.text}</code>"

    return render_template("index.html", response=html_response)

if __name__ == "__main__":
    app.run(debug=True)
