docker build --tag=huggingface_flask_chatbox  --no-cache --progress=plain  . &> build.log

docker run -p 4321:4321 huggingface_flask_chatbox