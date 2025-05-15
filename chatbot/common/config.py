import pymysql

def connect_db(host='127.0.0.1', user='nlp', pwd='dongwon', db_name='chatbot_db'):
    conn = pymysql.connect(
        host=host,
        user=user,
        password=pwd,
        db=db_name,
        charset='utf8'
        )
    cur = conn.cursor() # 커서생성

    return conn, cur