function sendMessage(text, type) {
    $('.write_msg').val('');
    if (type == 'bot'){
        $('.msg_history').append('<div class="incoming_msg"><div class="incoming_msg_img"><img src="static/img/user-profile.png" alt="sunil"></div> <div class="received_msg"><div class="received_withd_msg"><p>'+text+'</p></div></div></div>')
    }
    else if(type == 'user'){
        $('.msg_history').append('<div class="outgoing_msg"><div class="sent_msg"><p>'+text+'</p></div></div>')
    }
    $('.msg_history').scrollTop(9999);
}

function requestChat(messageText, okay, dialog_node, node_detail, parent, condition, url_pattern) {
    // http://172.30.1.228:8080/
    $.ajax({
        url: "http://192.168.3.38:8080/" + url_pattern,
        type: "POST",
        dataType: "json",
        data: {
            'messageText': messageText,
            'okay': okay,
            'intent_no': intent_no,
            'node_detail': node_detail,
            'parent': parent,
            'condition': condition,
        },
        success: function (data) {
            if (data['type']=='bot'){
                if (Object.keys(data).includes('ending')){
                    ending_len = data['ending'].length
                    $.each(data['ending'], function(index, item){
                        setTimeout(function(){sendMessage(item['text'], 'bot')}, 1000*index)
                        if (index==ending_len-1){
                            $('.intent_no').attr('value', item['intent_no'])
                            $('.node_detail').attr('value', item['node_detail'])
                            $('.condition').attr('value', item['condition'])
                            $('.parent').attr('value', item['parent'])
                        }
                    })
                } else{
                    sendMessage(data['text'], 'bot');
                    $('.condition').attr('value', data['condition'])
                }

                if (($('.condition').val()=='YN') || ($('.condition').val()=='ABCD') || ($('.condition').val()=='intent')){
                    $('.okay').attr('value', 1)
                } else if ($('.condition').val()=='END'){
                    $('.okay').attr('value', 0)
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
    let intent_no = $('.intent_no').val();
    let node_detail = $('.node_detail').val();
    let parent = $('.parent').val();
    let condition = $('.condition').val();
    if (messageText.length!=0)
        sendMessage(messageText, 'user');
        button_obj = $('.msg_history').children('.incoming_msg').last().find('button')
        button_length = $('.msg_history').children('.incoming_msg').last().find('button').length
        button_text_list = []
        if (button_length >= 2){
            button_obj.each(function(index, item){
                button_text_list.push(item.textContent.replace('\n',''))
            })
            find_idx = button_text_list.indexOf(messageText)
            if (find_idx==-1){
                // 버튼에 들어있는 값이 아닐때
                okay = 0
                requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
            } 
            else {
                // 버튼에 들어있는 값일때
                if (condition=='YN'){
                    if (find_idx==0){
                        messageText = '네'
                    } else {
                        messageText = '아니오'
                    }
                    requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
                } else if (condition=='ABCD'){
                    let alphabet = find_idx + 65
                    let ABCD = String.fromCharCode(alphabet)
                    requestChat(ABCD, okay, intent_no, node_detail, parent, condition, 'request_chat');
                } else if (condition=="intent"){
                    const INTENT = btn_list[find_idx]["value"].toString()
                    console.log(INTENT)
                    messageText = e.target.textContent
                    sendMessage(messageText, 'user');
                    requestChat(INTENT, okay, dialog_node, node_detail, parent, condition, 'request_chat');
                } // test
                
            }
        } else{
            if (condition=='YN'){
                yno_list = ['네', '아니오']
                find_idx = yno_list.indexOf(messageText)
                if (find_idx==-1){
                    okay = 0
                    requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');    
                } else {
                    requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
                }
                
            } else {
                requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
            }
            
        }
}

function onClickAsEnter(e) {
    if (e.keyCode === 13 && $('.write_msg').val().length!=0) {
        onSendButtonClicked()
    }
}

$(document).on('click', 'button', function(e){
    let okay = $('.okay').val();
    let intent_no = $('.intent_no').val();
    let node_detail = $('.node_detail').val();
    let parent = $('.parent').val();
    let condition = $('.condition').val();
    const nodes = [...e.target.parentElement.children];
    btn_list = []
    console.log(e.target.className)
    $.each(nodes, function(index, item){
        if (item.tagName=='BUTTON'){
            btn_list.push(item)
        }
    })
    console.log(btn_list)
    const index = btn_list.indexOf(e.target);
    console.log(btn_list[index]["value"])
    if (e.target.onclick){
        // 웹페이지 이동
    } else {
        if (condition=='YN'){
            messageText = e.target.textContent
            sendMessage(messageText, 'user');
            if (index==0){
                let YNO = '네'
                requestChat(YNO, okay, intent_no, node_detail, parent, condition, 'request_chat');
            } else {
                let YNO = '아니오'
                requestChat(YNO, okay, intent_no, node_detail, parent, condition, 'request_chat');
            }
        } else if (condition=='ABCD'){
            let alphabet = index + 65
            let ABCD = String.fromCharCode(alphabet)
            messageText = e.target.textContent
            sendMessage(messageText, 'user');
            requestChat(ABCD, okay, intent_no, node_detail, parent, condition, 'request_chat');
        } else if (e.target.className=='dial_btn'){
            let intent_no = btn_list[index]["value"].toString()
            console.log(intent_no)
            messageText = e.target.textContent
            sendMessage(messageText, 'user');
            requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
        }
    }   
})