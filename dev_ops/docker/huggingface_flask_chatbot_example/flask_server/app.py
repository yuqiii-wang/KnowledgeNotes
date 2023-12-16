from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session, sessions
from flask_cors import CORS, cross_origin

from llm_pipe_models import LLMModels

llmModels = LLMModels()

app = Flask(__name__, static_folder="static")
app.config['CORS_HEADERS'] = 'Content-Type'
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)
app.secret_key = 'abcdefghi'


@app.route('/', methods = ['GET'])
@app.route('/index', methods = ['GET'])
@app.route('/home', methods = ['GET'])
def home():
   session["used_model_name"] = "t5_small"
   return render_template("home.html", model_names=llmModels.getAllModelNames(), used_model_name=session["used_model_name"])
   

@app.route('/switchLLM', methods = ['POST'])
def switchLLM():
   print("switchLLM: " + request.form['payload'])
   session["used_model_name"] = request.form['payload']
   return jsonify(render_template("switchModel.html", used_model_name=session["used_model_name"]))
   

@app.route('/', methods = ['POST', 'OPTIONS'])
@cross_origin(origin='127.0.0.1', headers=['Content-Type','Authorization'])
def processChat():
   chatContext = []
   print(session["used_model_name"])
   if request.method == 'POST':
      chatContext.append(request.form['payload'])
      print(request.form['payload'])
      model = llmModels.getModel(session["used_model_name"])
      output = model(request.form['payload'])
      print(session["used_model_name"] + ": " + str(output))
      chatContext.append(output[0]["generated_text"])
      return jsonify(render_template("chats.html", chats = chatContext))
   return render_template("404.html")


@app.after_request
def handle_options(response):
    response.headers["Access-Control-Allow-Origin"] = "http://127.0.0.1:4321"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    response.headers['Vary']='Origin'
    return response

if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0', port=4321)
