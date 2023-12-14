from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_session import Session, sessions
from flask_cors import CORS, cross_origin

from transformers import pipeline
from transformers import AutoConfig, AutoModelForSeq2SeqLM, PeftModelForSeq2SeqLM, get_peft_config


# config_lora = {
#     "peft_type": "LORA",
#     "task_type": "SEQ_2_SEQ_LM",
#     "inference_mode": False,
#     "r": 16,
#     "lora_alpha": 32,
#     "lora_dropout": 0.05,
#     "fan_in_fan_out": False,
#     "bias": "none",
# }
# model = AutoModelForSeq2SeqLM.from_pretrained("llm_models/t5-small-lora-sum/t5-small")
# peft_config = get_peft_config(config_lora)
# model_checkpoint = AutoModelForSeq2SeqLM.from_pretrained("./t5-small-lora/checkpoint-920")
# peft_model_lora = PeftModelForSeq2SeqLM(model_checkpoint, peft_config)

pipe_flan = pipeline("text2text-generation", model="llm_models/google/flan-t5-small")
pipe_t5_small = pipeline("text2text-generation", model="llm_models/t5-small")

pipe_model = dict()
pipe_model["flan"] = pipe_flan
pipe_model["t5_small"] = pipe_t5_small


app = Flask(__name__, static_folder="static")
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = b'abcdefghi'


@app.route('/', methods = ['GET'])
@app.route('/index', methods = ['GET'])
@app.route('/home', methods = ['GET'])
def home():
   if not "used_model_name" in session.keys():
      session["used_model_name"] = "t5_small"
   return render_template("home.html", model_names=pipe_model.keys(), used_model_name=session["used_model_name"])
   

@app.route('/switchLLM', methods = ['POST'])
def switchLLM():
   print(request.form['payload'])
   session["used_model_name"] = request.form['payload'].lower()
   # return redirect(url_for('home'))
   return jsonify(render_template("switchModel.html", used_model_name=session["used_model_name"]))
   

@app.route('/', methods = ['POST', 'OPTIONS'])
@cross_origin(origin='127.0.0.1', headers=['Content-Type','Authorization'])
def processChat():
   chatContext = []
   if request.method == 'POST':
      chatContext.append(request.form['payload'])
      print(request.form['payload'])
      output = pipe_model[session["used_model_name"]](request.form['payload'])
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
   app.run(debug = True, port=4321)
