from flask import Flask, request
from video_text import analyze_video
from train_model import classify_image

app = Flask(__name__)

@app.route('/classify', methods=['GET'])
def classify():
    video_url = request.args.get('video_url')
    app.logger.debug(f'Received video_url: {video_url}')
    if video_url:
        result = analyze_video(video_url)
        return f"{result}"
    else:
        return "Error: No video_url provided", 400

@app.route('/image', methods=['GET'])
def classify_image_route():
    image_url = request.args.get('image_url')
    app.logger.debug(f'Received image_url: {image_url}')
    if image_url:
        result = classify_image(image_url)
        return f"{result}"
    else:
        return "Error: No image_url provided", 400

if __name__ == "__main__":
    app.run(debug=True)




# http://localhost:5000/classify?video_url=https%3A%2F%2Futfs.io%2Ff%2F978f9838-8ab3-4c8c-a1c8-efb41f609597-b08w7r.mp4
# http://localhost:5000/image?image_url=https://utfs.io/f/059547f8-fafc-482b-9b53-5e5dbefbc2e4-m2318w.jpeg