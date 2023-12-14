function displayModelList() {
    document.getElementById("llm_model_list").toggleAttribute("hidden");
}

function switchModel(val) {

    model_name = val;

    if (model_name == "") {
        console.log(model_name);
        return;
    }

    console.log(model_name);
    $.ajax({
        type:"POST",
        url:'http://localhost:4321/switchLLM',
        data: {payload: model_name},
        success:function(responsedata){
            // process on data
            document.getElementById("chat_title_model_name").innerHTML = "Chat with " + model_name;
            // $('div#chat_delimiter_model_name').append(model_name);
            console.log(responsedata);
            document.getElementById("llm_model_list").toggleAttribute("hidden");
            $('div#chats').append(responsedata);
        }
    })

};