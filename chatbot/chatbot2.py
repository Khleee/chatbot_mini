from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, default_exceptions, _aborter

import numpy as np
import pandas as pd
import random 
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
dialog_df = pd.read_csv("data/dialog2.csv", encoding='cp949')

B = pd.read_csv("data/title_node.csv")


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
def DIA(start):    
    # 첫번째 응답 메세지
    first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==start) & (dialog_df['node_detail']=='0')]
    random_number = random.randrange(0, len(first_msg_df))
    selected_first_msg = first_msg_df.iloc[random_number]
    
    response_list = []

    response_list.append({'dialog_node':int(selected_first_msg['dialog_node']), 
                          'node_detail':selected_first_msg['node_detail'], 
                          'text':selected_first_msg['text'], 
                          'parent':selected_first_msg['parent'], 
                          'condition':selected_first_msg['condition']})
    seq_filter = r'^(0)_+[0-9]{1,}$'
    print('selected_first_msg', selected_first_msg)
    if selected_first_msg['condition']=='seq':
        filter_df = dialog_df.loc[(dialog_df['dialog_node']==start) & dialog_df['node_detail'].str.match(seq_filter)==True]
        response_list = response_list + filter_df.to_dict('records')
        return response_list
    elif selected_first_msg['condition']=='END':
        return response_list
    elif selected_first_msg['condition']=='ABCD':
        return response_list
    elif selected_first_msg['condition']=='YNO':
        return response_list
        
# 2번째부터
def DIA2(messageText, dialog_node, node_detail, parent, condition):
    response_list = [] 
    if condition=='YNO':
        if messageText=='네':
            print('네')   
            node_detail = node_detail+'-Y'
            first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & (dialog_df['node_detail']==node_detail)]
            selected_first_msg = first_msg_df.iloc[0]
            response_list.append({'dialog_node':int(selected_first_msg['dialog_node']), 
                          'node_detail':selected_first_msg['node_detail'], 
                          'text':selected_first_msg['text'], 
                          'parent':selected_first_msg['parent'], 
                          'condition':selected_first_msg['condition']})
            if selected_first_msg.loc['condition']=='seq':
                seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                print('seq_filter', seq_filter)
                filter_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                response_list = response_list + filter_df.to_dict('records')
        elif messageText=='아니오':
            print('아니오')
            node_detail = node_detail+'-N'            
            first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & (dialog_df['node_detail']==node_detail)]
            if len(first_msg_df)==0:
                pass
            else:
                selected_first_msg = first_msg_df.iloc[0]
                response_list.append({'dialog_node':int(selected_first_msg['dialog_node']), 
                            'node_detail':selected_first_msg['node_detail'], 
                            'text':selected_first_msg['text'], 
                            'parent':selected_first_msg['parent'], 
                            'condition':selected_first_msg['condition']})
                if selected_first_msg.loc['condition']=='seq':
                    seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                    filter_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                    response_list = response_list + filter_df.to_dict('records')
        else:
            print('몰라')
            node_detail = node_detail+'-O'            
            first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & (dialog_df['node_detail']==node_detail)]
            # print(type(first_msg_df))
            # response_list = response_list + [{'dialog_node':dialog_node, 'node_detail':node_detail[:-2], 'text':first_msg_df.iloc[0]['text'], 'parent':first_msg_df.iloc[0]['parent'], 'condition':first_msg_df.iloc[0]['condition']}]
            # print('after response_list', response_list)
            response_list = response_list + first_msg_df.to_dict('records')
    elif condition=='ABCD':
        abcd_filter = r'^' + node_detail+'-[A-Z]{1}$'
        abcd_filter_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & dialog_df['node_detail'].str.match(abcd_filter)==True]
        print('abcd_filter_df', abcd_filter_df)
        abcd_list = abcd_filter_df['node_detail'].map(lambda x: x.split('-')[-1]).tolist()
        print('abcd_list', abcd_list)
        print('messageText', messageText)
        for abcd in abcd_list:
            if messageText == abcd:
                node_detail = node_detail + '-' + messageText
                first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & (dialog_df['node_detail']==node_detail)]
                selected_first_msg = first_msg_df.iloc[0]
                response_list.append({'dialog_node':int(selected_first_msg['dialog_node']), 
                                'node_detail':selected_first_msg['node_detail'], 
                                'text':selected_first_msg['text'], 
                                'parent':selected_first_msg['parent'], 
                                'condition':selected_first_msg['condition']})
                if selected_first_msg.loc['condition']=='seq':
                    seq_filter = r'^' + node_detail+'_[0-9]{1,}$'
                    filter_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & dialog_df['node_detail'].str.match(seq_filter)==True]
                    response_list = response_list + filter_df.to_dict('records')
    elif condition=='BACK':
        first_msg_df = dialog_df.loc[(dialog_df['dialog_node']==int(dialog_node)) & (dialog_df['node_detail']==node_detail)]
        selected_first_msg = first_msg_df.iloc[0]
        response_list.append({'dialog_node':int(selected_first_msg['dialog_node']), 
                        'node_detail':selected_first_msg['node_detail'], 
                        'text':selected_first_msg['text'], 
                        'parent':selected_first_msg['parent'], 
                        'condition':selected_first_msg['condition']})
    elif condition=='seq':
        pass
    else:
        pass
    if len(response_list)==0:
            response_list = response_list + [{'text':"답변 데이터가 없습니다. 처음으로 돌아갑니다.", 'type':'bot', 'okay':0, 'condition':'END'}]
    return response_list           
                
@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/request_chat', methods=['POST'])
def request_chat(): # enter치면
    messageText = request.form['messageText'] # 방금 메시지 입력한거 들어감
    okay = request.form['okay'] # 기존에 들어간 okay
    okay = int(okay)
    dialog_node = request.form['dialog_node'] 
    node_detail = request.form['node_detail'] 
    parent = request.form['parent'] 
    condition = request.form['condition'] 
    
    # print("ending :",ending)
    # ending = ending.split(',')
    #print("ending :",ending)

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
        pred_idx = -1

        for x in probs:
            if x >= 0.5:
                print("이해O",B["title"][np.argmax(probs)],np.argmax(probs), probs[np.argmax(probs)])
                pred_idx = np.argmax(probs)
                break
        
        if pred_idx == -1:
            print("이해X",B["title"][np.argmax(probs)],np.argmax(probs), probs[np.argmax(probs)])
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})

        start = pred_idx

        okay = 1
        print('의도번호', start)

        ending = DIA(start)
        print('okay 0 response', ending)
        if ending == None:
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
        else:
            return jsonify({'ending':ending, 'type':'bot'})
    
    elif okay==1:
        ending = DIA2(messageText, dialog_node, node_detail, parent, condition)
        print('okay 1 response', ending)
        if ending == None:
            return jsonify({'text':"이해하기 어려워요. 쉽게 얘기해주세요", 'type':'bot', 'okay':0})
        else:
            return jsonify({'ending':ending, 'type':'bot'})
    
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)