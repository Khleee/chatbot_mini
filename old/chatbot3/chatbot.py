from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, default_exceptions, _aborter

import numpy as np
import pandas as pd
import time
import random

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
A = pd.read_csv("data/dialog2_보기편하게_링크수정.csv",encoding="CP949")
# A = pd.read_csv("data/dialog2.csv")
# del A["intent"]

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
    if start2 == "나가기": # 나가기
        time.sleep(0.5)
        print("다시 돌아가겠습니다.")
        return None
    
    ending_all = []
    for x in A.iloc:
        if str(x["dialog_node"]) == str(start2) and str(x["node_detail"]) == str(0):
            ending_all.append([str(x["dialog_node"]),x["node_detail"],[x["text"]],x["condition"]])

    # print(ending_all)

    if ending_all == []:
        return None
    else:
        # print(random.choice(ending_all))
        return random.choice(ending_all)

            # ending_temp = [str(x["dialog_node"]),x["node_detail"],[x["text"]],x["condition"]]
            # return ending_temp
# 2번째부터
def DIA2(start2, ending1, ending2, ending3, ending4): # 왜인지는 몰라도 여기서 쪼개져버림 
    if start2 == "나가기":
        time.sleep(0.5)
        print("다시 돌아가겠습니다.")
        return None
    if ending4 == "YNO":
        # 그 다음 선택지도 선택지일수도 있고 아닐수도있음 -> 확인해서 [1]을 수정해놔야한다!
        if start2 == "네": # mrc 이용해서 intent : "긍정", 동의어 : ["응","맞아","그래"] 지정
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-Y")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp

        elif start2 == "아니요":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-N")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp

        else: # "몰라요": mrc 이용해서 잘못된 응답임을 확인하던가 ###
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-O")]
            if temp.empty == False:
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            else:
                ending_temp = [str(int(ending1)),ending2,["잘 이해하지 못했어요.. 다시 답변해주시면 감사하겠습니다."],ending4]
                print("몰라요",ending_temp)
            return ending_temp

    elif ending4 == "ABCD": # 버튼식으로 나중에 구현
        if start2 == "A":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-A")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp
        elif start2 == "B":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-B")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp
        elif start2 == "C":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-C")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp
        elif start2 == "D":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-D")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp
        elif start2 == "E":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-E")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp
        elif start2 == "F":
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"-F")]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
            return ending_temp



    elif ending4 == "seq":
        # print("여기까지 왔구나")
        if len(ending2) >= 3: # 여러 경로 타다 옴
            if ending2[-2] == "_":# _9까지만 동작함
                temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2[:-1]+str(int(ending2[-1])+1))]
                temp_temp = ending3 + [temp["text"].values[0]]
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp_temp,temp["condition"].values[0]]

                while temp.empty == False:
                    if ending_temp[1][-2] == "_":# _9까지만 동작함
                        temp = A[(A["dialog_node"] == int(ending_temp[0])) & (A["node_detail"] == ending_temp[1][:-1]+str(int(ending_temp[1][-1])+1))]
                        if temp.empty == True:
                            break
                        temp2 = ending_temp[2]+[temp["text"].values[0]]

                        ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp2,temp["condition"].values[0]]

                return ending_temp

            else: # 맨처음이라는 뜻
                temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"_0")]
                temp_temp = ending3 + [temp["text"].values[0]]
                ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp_temp,temp["condition"].values[0]]

                while temp.empty == False:
                    if len(ending_temp[1]) >= 3:
                        if ending_temp[1][-2] == "_":# _9까지만 동작함
                            temp = A[(A["dialog_node"] == int(ending_temp[0])) & (A["node_detail"] == ending_temp[1][:-1]+str(int(ending_temp[1][-1])+1))]
                            if temp.empty == True:
                                break
                            temp2 = ending_temp[2]+[temp["text"].values[0]]

                            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp2,temp["condition"].values[0]]
                return ending_temp

        else: # len(ending2) = 0, 1 인경우 => 처음 시작한다
            temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2+"_0")]
            temp_temp = ending3 + [temp["text"].values[0]]
            ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp_temp,temp["condition"].values[0]]
            while temp.empty == False:
                if len(ending_temp[1]) >= 3:
                    if ending_temp[1][-2] == "_":# _9까지만 동작함
                        temp = A[(A["dialog_node"] == int(ending_temp[0])) & (A["node_detail"] == ending_temp[1][:-1]+str(int(ending_temp[1][-1])+1))]
                        if temp.empty == True:
                            break
                        temp2 = ending_temp[2]+[temp["text"].values[0]]

                        ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],temp2,temp["condition"].values[0]]
            return ending_temp

    elif ending4 == "BACK":
        temp = A[(A["dialog_node"] == int(ending1)) & (A["node_detail"] == ending2[:-2])]
        ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],[temp["text"].values[0]],temp["condition"].values[0]]
        # print("몰라요2",ending_temp)
        return ending_temp

    # elif ending4 == "END":
    #     ending_temp = [str(temp["dialog_node"].values[0]),temp["node_detail"].values[0],"다시 처음으로 돌아가겠습니다.",temp["condition"].values[0]]
    #     return ending_temp

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

    # if ending4 == "END":
    #     return jsonify({'text':["다시 처음으로 돌아가겠습니다."], 'type':'bot', 'okay':0})
        # return jsonify({'ending1' : ending1,'ending2' : ending2, 'ending3' : ending3, 'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
    if (okay == 0) or (ending4 == "END") or (ending4 == "BACK"):
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

        ending = DIA(start)

        # print('ending', ending) # [x["dialog_node"],x["node_detail"],x["text"],x["condition"]]
        if ending == None:
            return jsonify({'text':["이해하기 어려워요. 쉽게 얘기해주세요"], 'type':'bot', 'okay':0})
        elif ending[3] == "seq":
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]

            ending = DIA2(messageText,ending1,ending2,ending3,ending4)
            # print("dkdkdk",ending)

            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]
            return jsonify({'ending1' : ending1,'ending2' : ending2,'ending3' : ending3,'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
        else:
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]
            return jsonify({'ending1' : ending1,'ending2' : ending2,'ending3' : ending3,'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
    
    elif okay==1:
        ending = DIA2(messageText,ending1,ending2,ending3,ending4)
        # print("dia2",ending)
        if ending == None:
            return jsonify({'text':["다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요"], 'type':'bot', 'okay':0})

        if ending[3] == "seq":
            # print("아니")
            ending = DIA2(messageText,ending[0],ending[1],ending[2],ending[3])

            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]
            return jsonify({'ending1' : ending1,'ending2' : ending2, 'ending3' : ending3, 'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})

        if messageText == "나가기":
            return jsonify({'text':["다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요"], 'type':'bot', 'okay':0})
        else:
            ending1 = ending[0]
            ending2 = ending[1]
            ending3 = ending[2]
            ending4 = ending[3]

            return jsonify({'ending1' : ending1,'ending2' : ending2, 'ending3' : ending3, 'ending4' : ending4, 'start':messageText, 'text':ending3, 'type':'bot', 'okay':1})
    
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=8080) # 포트만 바꾸면 됨