# App 스크립트 파일 작성
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug = True) # debug: 코드를 수정할 때마다 다시 시작