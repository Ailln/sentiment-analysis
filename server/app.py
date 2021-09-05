from flask import Flask
from flask import request

from app.test import inference

app = Flask(__name__)


@app.route("/sentimentAnalysis")
def index():
    text = request.args.get("text")
    print(text)
    return {"text": inference(text)}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8012)
    # curl http://127.0.0.1:8012/sentimentAnalysis?text=%E6%88%91%E5%BE%88%E5%BC%80%E5%BF%83
