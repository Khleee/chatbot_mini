import pymysql

def connect_db(host='172.30.1.204', port=3306, user='nlp', pwd='dongwon', db_name='chatbot_db'):
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        passwd=pwd,
        db=db_name,
        charset='utf8'
        )
    cur = conn.cursor() # 커서생성

    return conn, cur