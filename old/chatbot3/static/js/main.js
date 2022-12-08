function sendMessage(text, type) {
    $('.write_msg').val('');
    if (type == 'bot'){
        $('.msg_history').append('<div class="incoming_msg"><div class="incoming_msg_img"><img src="https://ptetutorials.com/images/user-profile.png" alt="sunil"></div> <div class="received_msg"><div class="received_withd_msg"><p>'+text+'</p><span class="time_date">11:01 AM    |    June 9</span></div></div></div>')
    }
    else if(type == 'user'){
        $('.msg_history').append('<div class="outgoing_msg"><div class="sent_msg"><p>'+text+'</p><span class="time_date">11:01 AM    |    June 9</span></div></div>')
    }
    $('.msg_history').scrollTop(99999999);
}
function requestChat(messageText, okay, ending1,ending2,ending3,ending4, start, url_pattern) {
    $.ajax({
        url: "http://172.30.1.228:8080/" + url_pattern, //외부서버로 바꾸면 됨 http://59.10.188.211:8081/
        type: "POST",
        dataType: "json",
        data: {
            'messageText': messageText,
            'okay': okay,
            'ending1': ending1,
            'ending2': ending2,
            'ending3': ending3,
            'ending4': ending4,
            'start': start
        },
        success: function (data) {
            console.log(data)
            if (data['type']=='bot'){
                for (var x=0; x<data["text"].length; x++){
                        console.log(data['text'][x])
                        sendMessage(data['text'][x], 'bot');
                        $('.okay').attr('value', data['okay'])
                        $('.ending1').attr('value', data['ending1'])
                        $('.ending2').attr('value', data['ending2'])
                        $('.ending3').attr('value', data['ending3'])
                        $('.ending4').attr('value', data['ending4'])
                        $('.start').attr('value', data['start'])
                }
            }
        },

        error: function (request, status, error) {
            console.log(error);
            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'bot');
        }
    });
}

function greet() {
    setTimeout(function () {
        return sendMessage("동원챗봇 데모에 오신걸 환영합니다.", 'bot');
    }, 100);
}

function onSendButtonClicked() {
    let messageText = $('.write_msg').val();
    let okay = $('.okay').val();
    let ending1 = $('.ending1').val();
    let ending2 = $('.ending2').val();
    let ending3 = $('.ending3').val();
    let ending4 = $('.ending4').val();
    let start = $('.start').val();
    if (messageText.length!=0)
        sendMessage(messageText, 'user');
        requestChat(messageText, okay, ending1,ending2,ending3,ending4, start, 'request_chat');
    }

function onClickAsEnter(e) {
    if (e.keyCode === 13 && $('.write_msg').val().length!=0) {
        onSendButtonClicked()
    }
}