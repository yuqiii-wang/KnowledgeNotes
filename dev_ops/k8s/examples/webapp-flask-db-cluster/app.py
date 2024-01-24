from flask import Flask
from db import User

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/twice')
def hello_world2():
    return 'Hello, World! again!'

## By default, port: 5000
if __name__ == "__main__":
    app.run(debug=True)
