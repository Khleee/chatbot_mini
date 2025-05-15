// main.js
function sendMessage(text, type) {
    const msgElem = document.createElement("div");
    msgElem.classList.add("chat-msg", type);
    msgElem.innerHTML = text;  // ✅ HTML 태그 해석 가능
    document.querySelector(".chat-body").appendChild(msgElem);
    scrollToBottom();
}

function showLoading() {
    const loading = document.createElement("div");
    loading.classList.add("chat-msg", "bot");
    loading.id = "loading";
    loading.innerHTML = `
        <div class="loading-dots">
            <span></span><span></span><span></span>
        </div>`;
    document.querySelector(".chat-body").appendChild(loading);
    scrollToBottom();
}

function hideLoading() {
    const loading = document.getElementById("loading");
    if (loading) loading.remove();
}

function scrollToBottom() {
    const chatBody = document.querySelector(".chat-body");
    chatBody.scrollTop = chatBody.scrollHeight;
}

function requestChat(messageText, okay, intent_no, node_detail, parent, condition, url_pattern) {
    showLoading();
    $.ajax({
        url: '/' + url_pattern,
        type: 'POST',
        dataType: 'json',
        data: {
            'messageText': messageText,
            'okay': okay,
            'intent_no': intent_no,
            'node_detail': node_detail,
            'parent': parent,
            'condition': condition,
        },
        success: function (data) {
            hideLoading();
            if (data['type'] === 'bot') {
                if (data['ending']) {
                    data['ending'].forEach((item, index) => {
                        setTimeout(() => sendMessage(item['text'], 'bot'), 1000 * index);
                    });
                } else {
                    sendMessage(data['text'], 'bot');
                }
            }
        },
        error: function () {
            hideLoading();
            sendMessage("죄송합니다. 서버 연결에 실패했습니다.", "bot");
        }
    });
}

function onSendButtonClicked() {
    const messageText = document.querySelector(".write_msg").value;
    if (!messageText.trim()) return;
    document.querySelector(".write_msg").value = "";
    sendMessage(messageText, "user");

    const okay = document.querySelector(".okay").value;
    const intent_no = document.querySelector(".intent_no").value;
    const node_detail = document.querySelector(".node_detail").value;
    const parent = document.querySelector(".parent").value;
    const condition = document.querySelector(".condition").value;

    requestChat(messageText, okay, intent_no, node_detail, parent, condition, 'request_chat');
}

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    sendMessage("동원챗봇 데모에 오신걸 환영합니다.", "bot");
});
