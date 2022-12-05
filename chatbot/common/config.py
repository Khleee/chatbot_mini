import pymysql

## db 불러오기 172.30.1.204 // 192.168.3.37
def connect_db(host='192.168.3.37', user='nlp', pwd='dongwon', db_name='chatbot_db'):
    conn = pymysql.connect(
        host=host,
        user=user,
        password=pwd,
        db=db_name,
        charset='utf8'
        )
    cur = conn.cursor() # 커서생성

    return conn, cur