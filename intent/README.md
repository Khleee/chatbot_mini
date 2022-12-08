## 인텐트 예측 모델 목록

1. intent classification-LSTM.ipynb : LSTM 기반
2. bert2.ipynb : KoBert 기반
3. intent_cls_CNN : CNN 기반

## 그외

1. data 폴더 : 동원쪽에서 전달받은 데이터셋 모음
2. tokenization_kobert.py : KoBert 기반 모델에서 토크나이저로 사용되는 파일
3. cc.ko.300.bin : 프리트레인된 한국어 임베딩(fasttext)
   => nas 서버에 업로드 (\데이터사업추진팀\99. 개인폴더\설진영)

## 사용하는 것들
1. intent classification-LSTM.ipynb : LSTM 기반
   > ~~~
2. bert2.ipynb :
   > 토크나이저 : tokenization_kobert.py  
   > 데이터 : data/동원몰 챗봇_☆상용서버☆Live-dialog.json

   > 모델 : \\Kict-nas\데이터사업추진팀\99. 개인폴더\유민석\2022_동원챗봇\intents_모델\kobert 사용