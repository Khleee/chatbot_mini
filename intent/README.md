## 인텐트 예측 모델 목록
1. intent classification-LSTM.ipynb : LSTM 기반
2. bert2.ipynb : KoBert 기반
3. CNN_intent_colab : CNN 기반
4. bert4LM_2차개선_newdata_미리분리_git.ipynb : KoBert-lm 기반


## 그외
1. data 폴더 : 동원쪽에서 전달받은 데이터셋 모음
2. tokenization_kobert.py : KoBert 기반 모델에서 토크나이저로 사용되는 파일
3. cc.ko.300.bin : 프리트레인된 한국어 임베딩(fasttext)
   => nas 서버에 업로드 (\데이터사업추진팀\99. 개인폴더\설진영)

## 사용하는 데이터
1. intent classification-LSTM.ipynb :
   > ~~~
2. bert2.ipynb :
   > 토크나이저 : tokenization_kobert.py  
   > 데이터 : data/동원몰 챗봇_☆상용서버☆Live-dialog.json

   > 모델 : \\Kict-nas\데이터사업추진팀\99. 개인폴더\유민석\2022_동원챗봇\intents_모델\kobert 사용
3. bert4LM_2차개선_newdata_미리분리_git.ipynb :
   > 토크나이저 : tokenization_kobert.py  
   > 데이터 : data/final2_test2.csv || data/final2_train2.cvs || data/final2_valid2.csv 
   > 모델 : 위치 미정