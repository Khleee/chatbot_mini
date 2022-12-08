from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, default_exceptions, _aborter

import numpy as np
import pandas as pd
import time

import torch
from tokenization_kobert import KoBertTokenizer
from transformers import AutoModelForSequenceClassification

app = Flask(__name__)


## 모델 불러오기
# 모델 load
model = AutoModelForSequenceClassification.from_pretrained("model/kobert2").cuda()

# 토크나이저 선언
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

# load 파라미터(변경시 꼭 수정해야해!)
max_len = 124

# gpu 메모리 청소
torch.cuda.empty_cache()

# device = gpu 설정
if torch.cuda.is_available():     
    device = torch.device("cuda")
    print('%d개 존재함' % torch.cuda.device_count())
    print('사용할 GPU : ', torch.cuda.get_device_name(0))
else:
    print('GPU 없어서 CPU로 설정')
    device = torch.device("cpu")

## 파일 불러오기
A = pd.read_csv("data/dialog2.csv")
del A["intent"]

B = pd.read_csv("data/title_node.csv")
intent_list = list(B["title"])


# 모든 에러에 대해서 JSON 응답을 보낼 수 있게 등록
def error_handling(error):
    if isinstance(error, HTTPException):
        result = {
            'code' : error.code,
            'description' : error.description,
            'message' : str(error.code) + ' ' + error.name
        }
    resp = jsonify(result)
    resp.status_code = result['code']
    return resp
    
for code in default_exceptions.keys():
    app.register_error_handler(code, error_handling)

## 함수 선언
# 맨 처음
def DIA(start2):
    if start2 == "처음으로": # 처음이면
        time.sleep(0.5)
        print("다시 돌아가겠습니다.")
        return None
        
    for x in A.iloc:
        if str(x["dialog_node"]) == str(start2):
            ending_temp = [str(x["dialog_node"]),x["node_detail"],x["text"],x["condition"]]
            return ending_temp
# 2번째부터
def DIA2(start2,ending1, ending2, ending3, ending4): # 왜인지는 몰라도 여기서 쪼개져버림 
    if start2 == "처음으로":
        time.sleep(0.5)
        print("다시 돌아가겠습니다.")
        return None
    if ending4 == "YNO":
        # 그 다음 선택지도 선택지일수도 있고 아닐수도있음 -> 확인해서 [1]을 수정해놔야한다!
        if start2 == "네":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-Y")]
            print(temp)
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
            return ending_temp

        elif start2 == "아니요":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-N")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
            return ending_temp

        else: # "몰라요": mrc 이용해서 잘못된 응답임을 확인하던가 ###
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-O")]
            if temp != None:
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
            else:
                ending_temp = [str(int(ending1)),ending2,"잘 이해하지 못했어요.. 다시 답변해주시면 감사하겠습니다.",ending4]
            return ending_temp

    elif ending4 == "ABCD":
        pass # 나중에 구현

    elif ending4 == "seq":
        if len(ending2) >= 3:
            if ending2[-2] == "_":# _9까지만 동작함
                temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2[:-1]+str(int(ending2[-1])+1))]
                print("temp :",temp)
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
                return ending_temp

            else:
                temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"_0")]
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
                return ending_temp

        else: # len(ending2) = 0, 1 인경우 => 처음 시작한다
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"_0")]
            print(temp)
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
            return ending_temp

    elif ending4 == "BACK":
        temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2[:-2])]
        ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp["text"].values[0],temp["condition"].values[0]]
        return ending_temp

    elif ending4 == "END":
        ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],"다시 처음으로 돌아가겠습니다.",temp["condition"].values[0]]

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/request_chat', methods=['POST'])
def request_chat(): # enter치면
    messageText = request.form['messageText'] # 방금 메시지 입력한거 들어감
    okay = request.form['okay'] # 기존에 들어간 okay
    okay = int(okay)
    ending1 = request.form['ending1'] # 기존에 들어간 dialog_node
    ending2 = request.form['ending2'] # 기존에 들어간 node_detail
    ending3 = request.form['ending3'] # 기존에 들어간 text
    ending4 = request.form['ending4'] # 기존에 들어간 condition
    start = request.form['start'] # 챗봇이 답변하는 내용
    # print("ending :",ending)
    # ending = ending.split(',')
    #print("ending :",ending)

    if okay==0:
        # 토큰화 하기
        input_ids = []
        attention_masks = []

        ## 읽고 토큰화하기
        encoded_dict = tokenizer.encode_plus(
                                messageText,            
                                add_special_tokens = True,
                                max_length = max_len,
                                pad_to_max_length = True, # padding =True 나 padding="longest" 같은걸로 대체하세요
                                return_attention_mask = True,
                                return_tensors = 'pt')
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        ## 모델 추론
        model.eval()
        
        predictions = []
        last_layer_attentions = []

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        for i in range(len(input_ids)):
            ids = input_ids[i].unsqueeze(0)
            masks = attention_masks[i].unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids=ids,
                                attention_mask=masks)

            logits = outputs[0]
            logits = torch.softmax(logits,dim=1)
            last_layer_attention = outputs[1][-1]

            logits = logits.detach().cpu().numpy()
            last_layer_attention = last_layer_attention.detach().cpu().numpy()

            last_layer_attentions.append(last_layer_attention) 
            predictions.append(logits)

        probs = predictions[0][0]
        pred_idx = np.argmax(probs)
        start = pred_idx

        okay = 1
        #print('의도번호', start)

        ending = DIA(start)

        #print('ending', ending) # [x["dialog_node"],x["node_detail"],x["text"],x["condition"]]
        if ending == None:
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
        else:
            # 여기서 문제 발생
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]
            return jsonify({'ending1' : ending1,'ending2' : ending2,'ending3' : ending3,'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
    
    elif okay==1:
        # print("ending_second:",ending)
        ending = DIA2(messageText,ending1,ending2,ending3,ending4)
        print(ending)
        if messageText == "처음으로":
            return jsonify({'text':"다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요", 'type':'bot', 'okay':0})
        # elif ending == None:
        #     return jsonify({'text':"다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요", 'type':'bot', 'okay':0})
        elif ending[3] == "END":
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]
            
            return jsonify({'ending1' : ending1,'ending2' : ending2, 'ending3' : ending3, 'ending4' : ending4, 'start':messageText, 'text':'다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요', 'type':'bot', 'okay':0})
        
        else:
            # print("ending_last:",ending)
            # print("ending :",type(ending))
            # print("message :",type(messageText))
            # print("ending0 :",type(ending[0]))
            # print("ending1 :",type(ending[1]))
            # print("ending2 :",type(ending[2]))
            # print("ending3 :",type(ending[3]))
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]

            return jsonify({'ending1' : ending1,'ending2' : ending2, 'ending3' : ending3, 'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
    
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)