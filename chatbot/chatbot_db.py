from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, default_exceptions, _aborter

import numpy as np
import pandas as pd
import random

from datetime import datetime
import torch
from tokenization_kobert import KoBertTokenizer
from transformers import AutoModelForSequenceClassification
from common.config import connect_db

app = Flask(__name__)

## 모델 불러오기
# 모델 load // model/kobert2 // kobertLM_new_3_epoch23
model = AutoModelForSequenceClassification.from_pretrained("model/kobertLM_new_5").cuda()

# 토크나이저 선언
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert-lm")

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

# intent 리스트 뽑기
def get_intent_list():
    conn, cur = connect_db()
    cur.execute("SELECT * FROM chatbot_db.intent")
    dialog = cur.fetchall()
    conn.close()
    dialog_df = pd.DataFrame(dialog, columns=['intent_no','intent_name', 'description'])

    intent_list = list(dialog_df["intent_name"].values)
    return intent_list

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

def dup_check(response_list):
    ## 중복아니면 그대로, node_detail이 중복인 경우만 random.choice()
    response_list2 = []
    temp = []
    if len(response_list) > 1: # 아래도 똑같이 붙여넣으려고 >1 로 적용함
        for i, x in enumerate(response_list):
            if i == 0:
                dup = x["node_detail"]
                temp.append(x)
            else:
                if dup == x["node_detail"]:
                    temp.append(x)
                else:
                    response_list2.append(random.choice(temp))
                    temp = []
                    temp.append(x)
                    dup = x["node_detail"]
        if len(temp) > 1:
            temp = random.choice(temp)
            response_list2.append(temp)
            return response_list2

        response_list2 += temp
        # print(response_list2)

        return response_list2
    else:
        return response_list

## 함수 선언
# 맨 처음
def DIA(start):
    """
    다이얼로그를 처음부터 시작하는경우만 동작하는 함수    
    response_list에 다양한 인자들을 채워 넣어 return 해준다

    start : intent에 해당하는 숫자(인덱스)
     
    """
    conn, cur = connect_db()
    cur.execute("SELECT * FROM dialog")
    dialog = cur.fetchall()
    conn.close()
    dialog_df = pd.DataFrame(dialog, columns=['id', 'intent_no', 'node_detail', 'text', 'parent', 'condition'])
    
    dialog_df.drop(['id'], axis=1, inplace=True)
    # 첫번째 응답 메세지
    first_msg_df = dialog_df.loc[(dialog_df['intent_no']==int(start)) & (dialog_df['node_detail']=='0')]
    random_number = random.randrange(0, len(first_msg_df))
    selected_first_msg = first_msg_df.iloc[random_number]
    
    response_list = []

    response_list.append({'intent_no':int(selected_first_msg['intent_no']),#! dialog_node -> intent_no 
                          'node_detail':selected_first_msg['node_detail'], 
                          'text':selected_first_msg['text'], 
                          'parent':selected_first_msg['parent'], 
                          'condition':selected_first_msg['condition']})
    seq_filter = r'^(0)_+[0-9]{1,}$'
    print('selected_first_msg', selected_first_msg)
    if selected_first_msg['condition']=='SEQ':
        filter_df2 = dialog_df.loc[(dialog_df['intent_no']==start) & dialog_df['node_detail'].str.match(seq_filter)==True].copy()
        # fileter_df2 요소들의 키 통일 (dialog_node)
        filter_df2.rename(columns = {'intent_no' : 'dialog_node'}, inplace = True) #!! intent_no -> dialog_node로 키 이름 수정
        response_list = response_list + filter_df2.to_dict('records')
        
        return dup_check(response_list)

    return dup_check(response_list)

# 2번째부터
def DIA2(messageText, intent_no, node_detail, parent, condition):
    """
    다이얼로그 처음 동작 이후에 계속 동작하는 함수    
    response_list에 다양한 인자들을 채워 넣어 return 해준다

    messageText : 사용자 응답 텍스트
    intent_no : 다이얼로그 숫자(인덱스)
    node_detail : 다이얼로그 세부 노드(ex) 0_5, 0-Y-N)
    parent : default가 기본값, 뒤로 돌아갈때, 단순히 이전 다이얼로그가 아닌 다른 다이얼로그로 연결 가능케함
    condition : 현재 노드의 종류 (YN, ABCD, intent, ...)     
    """
    conn, cur = connect_db()
    cur.execute("SELECT * FROM dialog")
    dialog = cur.fetchall()
    conn.close()
    dialog_df = pd.DataFrame(dialog, columns=['id', 'intent_no', 'node_detail', 'text', 'parent', 'condition'])
    dialog_df.drop(['id'], axis=1, inplace=True)
    response_list = []

    if condition=='YN':
        if messageText=='네':
            print('네')
            node_detail = node_detail+'-Y'
            first_msg_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & (dialog_df['node_detail']==node_detail)]
            random_number = random.randrange(0, len(first_msg_df))
            selected_first_msg = first_msg_df.iloc[random_number]
            response_list.append({'intent_no':int(selected_first_msg['intent_no']), 
                          'node_detail':selected_first_msg['node_detail'], 
                          'text':selected_first_msg['text'], 
                          'parent':selected_first_msg['parent'], 
                          'condition':selected_first_msg['condition']})
            if selected_first_msg.loc['condition']=='SEQ':
                seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                print('seq_filter', seq_filter)
                filter_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                response_list = response_list + filter_df.to_dict('records')
        elif messageText=='아니오':
            print('아니오')
            node_detail = node_detail+'-N'            
            first_msg_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & (dialog_df['node_detail']==node_detail)]
            random_number = random.randrange(0, len(first_msg_df))
            selected_first_msg = first_msg_df.iloc[random_number]
            response_list.append({'intent_no':int(selected_first_msg['intent_no']), 
                          'node_detail':selected_first_msg['node_detail'], 
                          'text':selected_first_msg['text'], 
                          'parent':selected_first_msg['parent'], 
                          'condition':selected_first_msg['condition']})
            if selected_first_msg.loc['condition']=='SEQ':
                seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                filter_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                response_list = response_list + filter_df.to_dict('records')
    elif condition=='ABCD':
        abcd_filter = r'^' + node_detail+'-[A-Z]{1}$'
        abcd_filter_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & dialog_df['node_detail'].str.match(abcd_filter)==True]
        print('abcd_filter_df', abcd_filter_df)
        abcd_list = abcd_filter_df['node_detail'].map(lambda x: x.split('-')[-1]).tolist()
        
        for abcd in abcd_list:
            if messageText == abcd:
                node_detail = node_detail + '-' + messageText
                first_msg_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & (dialog_df['node_detail']==node_detail)]
                random_number = random.randrange(0, len(first_msg_df))
                selected_first_msg = first_msg_df.iloc[random_number]
                response_list.append({'intent_no':int(selected_first_msg['intent_no']), 
                                'node_detail':selected_first_msg['node_detail'], 
                                'text':selected_first_msg['text'], 
                                'parent':selected_first_msg['parent'], 
                                'condition':selected_first_msg['condition']})
                if selected_first_msg.loc['condition']=='SEQ':
                    seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                    filter_df = dialog_df.loc[(dialog_df['intent_no']==int(intent_no)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                    response_list = response_list + filter_df.to_dict('records')

    elif condition=='SEQ':
        pass
    else:
        pass    
    return dup_check(response_list)

def DIA3(start, i_list): # 만약 i_list가 너무 많으면 선택지가 너무 많으므로, 개수 제한 걸어야됨
    """
    딥러닝 모델에서 인텐트를 파악하지 못하는 경우 동작함.
    이전 DIA 함수들과 달리, 유사 인텐트들을 출력해주는 역할

    구체적으로는 response_list에 임의로 text를 유사 인텐트들을 버튼식으로 출력되게 수정하고 있다.
    """
    
    """
    conn, cur = connect_db()
    cur.execute("SELECT * FROM dialog")
    dialog = cur.fetchall()
    conn.close()
    dialog_df = pd.DataFrame(dialog, columns=['id', 'intent_no', 'node_detail', 'text', 'parent', 'condition'])
    dialog_df.drop(['id'], axis=1, inplace=True)
    # 첫번째 응답 메세지
    first_msg_df = dialog_df.loc[(dialog_df['intent_no']==start) & (dialog_df['node_detail']=='0')]
    random_number = random.randrange(0, len(first_msg_df))
    selected_first_msg = first_msg_df.iloc[random_number]

    print('selected_first_msg', selected_first_msg)
    print('dia3에서 나온 최종 i_list:', i_list)
    
    """
    
    response_list = []

    select_list = ""

    for x in i_list:
        select_list += '<button class="dial_btn" value="' + str(x[0]) + '">'+str(x[1])+'</button>'

    # 현재 노드 상황
    # response_list.append({'intent_no':int(selected_first_msg['intent_no']), 
    #                       'node_detail':selected_first_msg['node_detail'], 
    #                       'text':selected_first_msg['text']+select_list, 
    #                       'parent':selected_first_msg['parent'], 
    #                       'condition':selected_first_msg['condition']})
    response_list.append({'text':"제대로 이해하지 못했어요. 이 중에서 하나 골라주세요.<br>"+select_list})
    return response_list
    
         
@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/request_chat', methods=['POST'])
def request_chat(): # enter치면
    intent_list = get_intent_list()
    messageText = request.form['messageText'] # 방금 메시지 입력한거 들어감
    okay = request.form['okay'] # 기존에 들어간 okay
    okay = int(okay)
    intent_no = request.form['intent_no'] 
    node_detail = request.form['node_detail'] 
    parent = request.form['parent']
    condition = request.form['condition']
    print("intent_no",intent_no)

    if messageText=='처음으로':
        return jsonify({'text':"다시 돌아가겠습니다. 문의사항이 있으시면 언제든 말씀해주세요", 'type':'bot', 'okay':0})
    
    if okay==0:
        # 토큰화 하기
        input_ids = []
        attention_masks = []

        ## 읽고 토큰화하기
        encoded_dict = tokenizer.encode_plus(
                                messageText,            
                                add_special_tokens = True,
                                max_length = max_len,
                                truncation = True, 
                                padding = "max_length",
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

        # # probs 높은순 확인용 코드
        # II = []
        # for i,x in enumerate(probs):
        #     II.append([i,x])

        # II.sort(key=lambda x: (-x[1], x[0]))
        # print("<높은순>")
        # print(II)

        ## 여기까지 intents 모델 끝

        # entity similar들어가서 messageText안에 존재하는지 확인
        # entity similar db 불러오기
        conn, cur = connect_db()

        cur.execute("SELECT * FROM entity_similar")
        dialog = cur.fetchall()
        conn.close()
        dialog_df2 = pd.DataFrame(dialog, columns=['similar_id','symbol_id', 'similar_name'])
        dialog_df2.drop(['similar_id'], axis=1, inplace=True)

        i_list = []
        for x,y in zip(dialog_df2["symbol_id"],dialog_df2["similar_name"]):
            if y in messageText: # 만약 전혀 연관없는 다른 엔티티에 각각 "어제는", "어제" 가 들어있다면, 걸러주지 못하고 그대로 i_list로 추가됨 그러므로, entity_similar에 다른 엔티티인데 비슷한 단어들이 들어있으면 안됨
                i_list.append(x) # 만약 속해있다면, i_list에 심볼id가 들어감

        ## 여기까지 입력문장에 해당 symbol_id가 뭐가 있는지 리스트화함
        print("i_list:",i_list)
        # entity_symbol들어가서 symbol_id에 해당하는 entity_id가 뭔지 확인
        conn, cur = connect_db()
        cur.execute("SELECT * FROM entity_symbol")
        dialog = cur.fetchall()
        conn.close()
        dialog_df2 = pd.DataFrame(dialog, columns=['symbol_id','entity_id', 'symbol_name'])
        
        i_list2 = []
        for x in i_list:
            i_list2 += list(dialog_df2[dialog_df2["symbol_id"]==x]["entity_id"].values)

        ## 여기까지 입력 문장안에 entity_id가 뭐가 들어있는지 확인함
        print("i_list2:",i_list2)
        # i_list2랑 인텐츠 안에 들어있는 엔티티 종류랑 비교할거임
        # intent_entity 들어가서 intent_no에 해당하는 entity_id 종류들을 꺼내기
        conn, cur = connect_db()
        cur.execute("SELECT * FROM intent_entity")
        dialog = cur.fetchall()
        conn.close()
        dialog_df2 = pd.DataFrame(dialog, columns=['id','entity_id', 'intent_no'])
        dialog_df2.drop(['id'], axis=1, inplace=True)

        i_list3 = []
        for x in list(dialog_df2["intent_no"]):
            if i_list2 == list(dialog_df2[dialog_df2["intent_no"]==x]["entity_id"]):
                conn, cur = connect_db()
                cur.execute("SELECT * FROM intent WHERE intent_no = %s", int(x))
                dialog = cur.fetchall()
                conn.close()
                i_list3.append(dialog[0])

        ## 여기까지 intents에 있는 entity_id와 입력 문장에 있는 entity_id들이 같을때의 intent_no을 저장함         
        print("i_list3:",i_list3)
        # 만약 일치하는 경우가 없으면, 그런 정보가 없다는 뜻이므로, 그대로 놔두면 됨
        # 만약 일치하는 경우가 하나 이상이면, 동작하자

        # intents 모델에서 나온 값을 이용하여 pred_idx -> 해당인덱스 -> start
        pred_idx = -1

        for x in probs:
            if x >= 0.5: # 50% 이상인 경우, argmax -> pred_idx
                print("이해O",intent_list[np.argmax(probs)],np.argmax(probs), probs[np.argmax(probs)]) #! B["title"] -> intent_list
                pred_idx = np.argmax(probs)
                break
        
        i_list4 = []
        intent_count = 1


        if pred_idx == -1: # 50% 이상인 경우가 아예 없었을 경우, 일단 20%~50% 사이의 인텐츠들을 뽑아보자
            """
            폴백 메시지에 대해서 데이터베이스에 고객이 발화한 대화와 발화한 날짜를 저장
            """
            conn, cur = connect_db()
            now = datetime.now()
            formatted_date = now.strftime('%Y-%m-%d')
            cur.execute("INSERT INTO fallback_message(message, fallback_date) values(%s, %s)", (messageText, formatted_date))
            conn.commit()
            conn.close()
            
            # 만약 i_list가 empty면 아예, 엔티티 정보가 없음-> 비슷한 인텐트를 출력해줄 수 없음
            if not i_list:
                return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})


            # 20% ~ 50% 인 intent(인덱스,확률값) 뽑기
            for i,x in enumerate(probs):
                if x > 0.2 and x < 0.5:
                    i_list4.append([i,x])
                    

            print("i_list4:",i_list4)

            # i_list3, i_list4(인텐트넘버) 간의 교집합 구하기
            real = list(set(i_list3) & set([x[0] for x in i_list4])) #! set(i_list4) -> set([x[0] for x in i_list4])
            print("real:",real)

            # real이 비워져있는일이 없는한 i_list4는 항상 존재
            # real2에 확률이 높은 순으로 채워놓음
            i_list4.sort(key=lambda x:-x[1])

            real2 = []
            for x in i_list4:
                if x[0] in real:
                    real2.append(x[0])

            
            if len(real) > 1:
                # 여러개 있으므로, new다이얼로그로 진입시켜서 인텐츠 선택하게 하자
                okay = 1
                ending = DIA3(251, real2[:4]) ## 여기다 real2 4개만 출력되게
                intent_count = 0

            elif len(real) == 1:
                # 사실상 하나만 있으므로, 바로 연결시켜주자
                okay = 1
                start = real[0]

                print('의도번호', start)
                ending = DIA(start)
                intent_count = 0

            elif len(real) == 0:
                # 겹치는게 없으므로, 인텐츠 동의어를 가진 intent 목록만 출력
                okay = 1
                # print('의도번호', start)
                ending = DIA3(251, i_list3[:4]) ## 여기다 i_list3 4개만 나오게
                # 아니면 그냥 20% 50% 사이에 있는 intents을 넣어도 될듯?(i_list4)
                intent_count = 0
                
        if intent_count == 1: # dia() 함수를 한번도 안거쳤으면 동작하게
            start = pred_idx # 50% 이상인 경우, start = pred_idx

            okay = 1
            print('의도번호', start)
            ending = DIA(start)

        print('okay 0 response', ending)
        print("messageText",messageText)
        if ending == None:
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
        else:
            return jsonify({'ending':ending, 'start':messageText, 'type':'bot'})
    
    elif okay==1:
        if condition == "intent":
            print('의도번호', messageText)
            ending = DIA(messageText)
            if ending == None:
                return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
            else:
                return jsonify({'ending':ending, 'start':messageText, 'type':'bot'})

        ending = DIA2(messageText, intent_no, node_detail, parent, condition)
        print('okay 1 response', ending)
        if ending == None:
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
        else:
            return jsonify({'ending':ending, 'start':messageText, 'type':'bot'})
    
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)