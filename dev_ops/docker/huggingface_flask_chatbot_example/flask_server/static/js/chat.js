function sendChatText() {

    //make an ajax call and get status value using the same 'id'
    var chatPrompts= document.getElementById("exampleFormControlInput1").value;

    if (chatPrompts == "") {
        console.log(chatPrompts);
        return;
    }

    console.log(chatPrompts);
    $.ajax({
        type:"POST",
        url:'http://localhost:4321/',
        data: {payload: chatPrompts},
        success:function(responsedata){
            // process on data
            if (responsedata == "" ) {
                responsedata = "..."
            }
            $('div#chats').append(responsedata);
        }
    })

};