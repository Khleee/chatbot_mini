import re
import pymysql
import os
from datetime import datetime
from common.model_user import User
from common.config import connect_db
from flask import Flask, redirect, render_template, request, url_for
from flask_paginate import Pagination, get_page_args
from flask_login import LoginManager, login_required, login_user, logout_user

app = Flask(__name__)
# 사용자 세션 관리 할 때 필요한 정보
app.secret_key = os.urandom(24)
# LoginManager object 생성
login_manager = LoginManager()
login_manager.init_app(app)

@app.route("/", methods=['GET'])
def root():
    return redirect('/login')

@app.route("/main", methods=['GET'])
def main():
    return render_template('main.html')

@app.route("/login", methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route("/login/get_info", methods=['GET', 'POST'])
def login_get_info():
    user_id = request.form.get('userID')
    user_password = request.form.get('userPassword')
    
    user_info = User.get_user_info(user_id, user_password)
    if user_info['result'] != 'fail':
        login_info = User(user_info['data'][0], user_info['data'][2], user_info['data'][5])
        login_user(login_info)
        print('login_info', login_info.get_id())
        return redirect('/main')
    else:
        login_result_text = '로그인 실패 메세지'
        return render_template('/login', login_result_text=login_result_text)

@login_manager.user_loader
def user_loader(user_id):
    user_info = User.get_user_info(user_id)
    login_info = User(user_info['data'][0], user_info['data'][2], user_info['data'][5])
    return login_info

@login_manager.unauthorized_handler
def unauthorized():
    # 로그인되어 있지 않은 사용자일 경우 첫화면으로 이동
    return redirect("/")

@app.route("/logout", methods=("GET",))
def logout():
    logout_user()
    return redirect('/')

@app.route("/defaultValue", methods=("GET",))
def default_value():
    conn, cur = connect_db()
    sql = ""
    sql += "SELECT intent_cut_off, similar_intent_score, intent_result_num, intent_try_num, b.user_id, b.user_name , info_date "
    sql += "FROM bot_info a, admin b "
    sql += "WHERE a.user_id = b.user_id"
    cur.execute(sql)
    result = cur.fetchone()
    conn.close()
    
    bot_info = {}
    bot_info['intent_cut_off'] = result[0]
    bot_info['similar_intent_score'] = result[1]
    bot_info['intent_result_num'] = result[2]
    bot_info['intent_try_num'] = result[3]
    bot_info['user_id'] = result[4]
    bot_info['user_name'] = result[5]
    bot_info['info_date'] = result[6].strftime('%Y-%m-%d %H:%M:%S')
    return render_template('default_value.html', bot_info=bot_info)

@app.route("/bot_info_update", methods=["POST"])
def bot_info_update():
    intent_cut_off = request.form.get('intent_cut_off')
    similar_intent_score = request.form.get('similar_intent_score')
    intent_result_num = request.form.get('intent_result_num')
    intent_try_num = request.form.get('intent_try_num')
    userID = request.form.get('userID')
    info_date = request.form.get('info_date')
    conn, cur = connect_db()
    sql = ""
    sql += "UPDATE bot_info "
    sql += "SET intent_cut_off=%s, similar_intent_score=%s, intent_result_num=%s, intent_try_num=%s, user_id=%s, info_date=%s;"
    cur.execute(sql, (intent_cut_off, similar_intent_score, intent_result_num, intent_try_num, userID, info_date))
    conn.commit()
    conn.close()
    return redirect('/defaultValue')
    
@app.route("/intent", methods=("GET",))  # index 페이지를 호출하면
def index():
    per_page = 10
    page, _, offset = get_page_args(per_page=per_page)  # 포스트 10개씩 페이지네이션을 하겠다.
    # 이 때 두 번째 return값은 per_page입니다.
    # 저는 per_page를 따로 get_page_args에 넣어줘서, per_page를 받아서 사용하지는 않았습니다.
    # page는 현재 위치한 page입니다. 기본적으로 1이고, 페이지 링크를 누르면 2, 3, ...입니다.
    # offset은 page에 따라 몇 번째 post부터 보여줄지입니다.
    # 기본적으로 0이고, 2페이지라면 10, 3페이지라면 20이겠죠.

    # DB 연결
    # conn = pymysql.connect(
    #     host='127.30.1.204',
    #     user='root',
    #     password='7557',
    #     db='chatbot_db',
    #     charset='utf8'
    #     )
    # cur = conn.cursor() # 커서생성
    conn, cur = connect_db()
    cur.execute("""
    SELECT COUNT(*)
    FROM (
    SELECT a.intent_no, a.intent_name, a.description, 
    (SELECT COUNT(intent_no) FROM intent_example b WHERE a.intent_no = b.intent_no) AS aa,
    (SELECT COUNT(intent_no) FROM dialog c WHERE a.intent_no = c.intent_no) AS bb
    FROM intent a
    GROUP BY a.intent_no
    ) AS cc
    """)  # 일단 총 몇 개의 포스트가 있는지를 알아야합니다.
    total = cur.fetchone()[0]
    cur.execute("""
    SELECT a.intent_no, a.intent_name, a.description, 
    (SELECT COUNT(intent_no) FROM intent_example b WHERE a.intent_no = b.intent_no) AS aa,
    (SELECT COUNT(intent_no) FROM dialog c WHERE a.intent_no = c.intent_no) AS bb
    FROM intent a
    GROUP BY a.intent_no ORDER BY a.intent_no ASC
                LIMIT %s OFFSET %s;
                """ %(per_page, offset))

    intents = cur.fetchall()
    conn.close()
    print(request.url)

    # print(dialogs)

    return render_template(
        "intent_temp.html",
        intents=intents,
        list=list,
        pagination=Pagination(
            page=page,  # 지금 우리가 보여줄 페이지는 1 또는 2, 3, 4, ... 페이지인데,
            total=total,  # 총 몇 개의 포스트인지를 미리 알려주고,
            per_page=per_page,  # 한 페이지당 몇 개의 포스트를 보여줄지 알려주고,
            prev_label="<<",  # 전 페이지와,
            next_label=">>",  # 후 페이지로 가는 링크의 버튼 모양을 알려주고,
            format_total=True,  # 총 몇 개의 포스트 중 몇 개의 포스트를 보여주고있는지 시각화,
        ),
        search=True,  # 페이지 검색 기능을 주고,
        bs_version=5,  # Bootstrap 사용시 이를 활용할 수 있게 버전을 알려줍니다.
    )

#################################### 추가 ####################################
@app.route("/write")  # index 페이지를 호출하면
def write():
    return render_template("write_page.html")

@app.route('/write_action', methods=['POST'])
def write_action():
    dialog = request.form.get('dialog')
    detail = request.form.get('detail')
    text = request.form.get('text')
    parent = request.form.get('parent')
    condition = request.form.get('condition')

    # conn = pymysql.connect(host='127.30.1.204', user='root', password='7557', db='chatbot_db', charset='utf8')
    # cur = conn.cursor()
    conn, cur = connect_db()
    SQL = "INSERT INTO dialog VALUES (NULL, %s, %s, %s, %s, %s);"
    values = (dialog, detail, text, parent, condition)
    cur.execute(SQL, values)
    conn.commit()
    conn.close()

    return redirect(url_for('index'))

#################################### 수정 ####################################
@app.route("/update", methods=['POST'])  # index 페이지를 호출하면
def update():
    did = request.form.get('did')

    conn, cur = connect_db()
    
    print(did)
    # SQL = 'SELECT * FROM dialog WHERE id=%d'
    # values = (did)
    # cur.execute(SQL, values)
    cur.execute('SELECT * FROM dialog WHERE id=%s' %did)
    dialog = cur.fetchone()
    conn.close()

    return render_template("update_page.html", dialog=dialog)

@app.route('/update_action', methods=['POST'])
def update_action():
    id = request.form.get('id')
    dialog = request.form.get('dialog')
    detail = request.form.get('detail')
    text = request.form.get('text')
    parent = request.form.get('parent')
    condition = request.form.get('condition')

    # print(type(id))
    # print(type(dialog))
    # print(type(detail))
    # print(type(text))
    # print(type(parent))
    # print(type(condition))

    # conn = pymysql.connect(host='127.30.1.204', user='root', password='7557', db='chatbot_db', charset='utf8')
    # cur = conn.cursor()
    conn, cur = connect_db()
    SQL = "UPDATE dialog SET intent_no=%s, node_detail=%s, text=%s, parent=%s, cdt=%s WHERE id=%s"
    values = (dialog, detail, text, parent, condition, int(id))
    cur.execute(SQL, values)
    # cur.execute("UPDATE dialog SET dialog_node = %s, node_detail = %s, text = %s, parent = %s, condition = %s, intent = %s WHERE id = %s" %(dialog, detail, text, parent, condition, intent, id))
    conn.commit()
    conn.close()

    return redirect(url_for('index'))

#################################### 삭제 ####################################
@app.route("/delete", methods=['POST'])  # index 페이지를 호출하면
def delete():
    did = request.form.get('did')

    conn, cur = connect_db()

    cur.execute('DELETE FROM dialog WHERE id=%s' %did)
    conn.commit()
    conn.close()

    return redirect(url_for('index'))

#################################### Entity list ####################################
@app.route("/entity", methods=("GET",))  # index 페이지를 호출하면
def entity():
    per_page = 20
    page, _, offset = get_page_args(per_page=per_page)  # 포스트 20개씩 페이지네이션을 하겠다.
    # 이 때 두 번째 return값은 per_page입니다.
    # 저는 per_page를 따로 get_page_args에 넣어줘서, per_page를 받아서 사용하지는 않았습니다.
    # page는 현재 위치한 page입니다. 기본적으로 1이고, 페이지 링크를 누르면 2, 3, ...입니다.
    # offset은 page에 따라 몇 번째 post부터 보여줄지입니다.
    # 기본적으로 0이고, 2페이지라면 10, 3페이지라면 20이겠죠.

    # DB 연결
    conn, cur = connect_db()

    cur.execute("SELECT COUNT(*) FROM entity;")  # 일단 총 몇 개의 포스트가 있는지를 알아야합니다.
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT * FROM entity ORDER BY entity_id ASC LIMIT %d OFFSET %d;" %(per_page, offset))  # SQL SELECT로 포스트를 가져오되,
        # "DESC LIMIT %s OFFSET %s;".format(per_page, offset)  # offset부터 per_page만큼의 포스트를 가져옵니다.

    entitys = cur.fetchall()
    conn.close()
    print(request.url)

    # print(dialogs)

    return render_template(
        "entity_temp.html",
        entitys=entitys,
        pagination=Pagination(
            page=page,  # 지금 우리가 보여줄 페이지는 1 또는 2, 3, 4, ... 페이지인데,
            total=total,  # 총 몇 개의 포스트인지를 미리 알려주고,
            per_page=per_page,  # 한 페이지당 몇 개의 포스트를 보여줄지 알려주고,
            prev_label="<<",  # 전 페이지와,
            next_label=">>",  # 후 페이지로 가는 링크의 버튼 모양을 알려주고,
            format_total=True,  # 총 몇 개의 포스트 중 몇 개의 포스트를 보여주고있는지 시각화,
        ),
        search=True,  # 페이지 검색 기능을 주고,
        bs_version=5,  # Bootstrap 사용시 이를 활용할 수 있게 버전을 알려줍니다.
    )

#################################### Entity 상세 조회 ####################################
@app.route("/entity_detail", methods=['POST'])  # index 페이지를 호출하면
def entity_detail():
    eid = request.form.get('eid')

    per_page = 20
    page, _, offset = get_page_args(per_page=per_page)  # 포스트 20개씩 페이지네이션을 하겠다.
    # 이 때 두 번째 return값은 per_page입니다.
    # 저는 per_page를 따로 get_page_args에 넣어줘서, per_page를 받아서 사용하지는 않았습니다.
    # page는 현재 위치한 page입니다. 기본적으로 1이고, 페이지 링크를 누르면 2, 3, ...입니다.
    # offset은 page에 따라 몇 번째 post부터 보여줄지입니다.
    # 기본적으로 0이고, 2페이지라면 10, 3페이지라면 20이겠죠.

    # DB 연결
    conn, cur = connect_db()

    cur.execute("SELECT COUNT(*) FROM entity_symbol WHERE entity_id = %s;" %(eid))  # 일단 총 몇 개의 포스트가 있는지를 알아야합니다.
    total = cur.fetchone()[0]

    cur.execute("SELECT * FROM entity_symbol WHERE entity_id = %s ORDER BY symbol_id ASC LIMIT %d OFFSET %d;" %(eid, per_page, offset))  # SQL SELECT로 포스트를 가져오되,
        # "DESC LIMIT %s OFFSET %s;".format(per_page, offset)  # offset부터 per_page만큼의 포스트를 가져옵니다.
    entity = cur.fetchall()
    print(entity[0][0])
    sym_id = []
    for e in entity:
        sym_id.append(e[0])

    print(entity)

    sim_list = []
    for i in sym_id:
        cur.execute("SELECT similar_name FROM entity_similar WHERE symbol_id = %s;" %(i))  # 일단 총 몇 개의 포스트가 있는지를 알아야합니다.
        sim_name_list = cur.fetchall()
        sim_name = []
        for n in sim_name_list:
            sim_name.append(re.sub('[,\(\)\']','',str(n)))    
        sim_list.append(sim_name)

    print(sim_list)

    print(request.url)

    # print(dialogs)

    return render_template(
        "entity_detail.html",
        entity = entity, 
        sim_list = sim_list,
        enumerate=enumerate,
        pagination=Pagination(
            page=page,  # 지금 우리가 보여줄 페이지는 1 또는 2, 3, 4, ... 페이지인데,
            total=total,  # 총 몇 개의 포스트인지를 미리 알려주고,
            per_page=per_page,  # 한 페이지당 몇 개의 포스트를 보여줄지 알려주고,
            prev_label="<<",  # 전 페이지와,
            next_label=">>",  # 후 페이지로 가는 링크의 버튼 모양을 알려주고,
            format_total=True,  # 총 몇 개의 포스트 중 몇 개의 포스트를 보여주고있는지 시각화,
        ),
        search=True,  # 페이지 검색 기능을 주고,
        bs_version=5,  # Bootstrap 사용시 이를 활용할 수 있게 버전을 알려줍니다.
    )

#################################### Entity 상세 조회 ####################################
@app.route("/fallback", methods=['GET'])  # index 페이지를 호출하면
def fallback():
    ### POST 방식
    # ymd = request.form.get('ymd')
    # print(ymd)

    ### GET 방식
    # ymd = request.args.to_dict()['ymd']
    # print(ymd)

    ### GET 방식
    ymd = request.args.get('ymd', datetime.now().strftime('%Y-%m-%d'))
    print(ymd)
    
    per_page = 10
    page, _, offset = get_page_args(per_page=per_page)  # 포스트 20개씩 페이지네이션을 하겠다.
    # 이 때 두 번째 return값은 per_page입니다.
    # 저는 per_page를 따로 get_page_args에 넣어줘서, per_page를 받아서 사용하지는 않았습니다.
    # page는 현재 위치한 page입니다. 기본적으로 1이고, 페이지 링크를 누르면 2, 3, ...입니다.
    # offset은 page에 따라 몇 번째 post부터 보여줄지입니다.
    # 기본적으로 0이고, 2페이지라면 10, 3페이지라면 20이겠죠.

    # DB 연결
    # conn = pymysql.connect(host='172.30.1.204', user='nlp', password='dongwoin', db='chatbot_db', charset='utf8')
    # cur = conn.cursor() # 커서생성
    conn, cur = connect_db()
    cur.execute("SELECT * FROM fallback_message WHERE fallback_date = '%s';" %(ymd))
    conn = pymysql.connect(host='172.30.1.204', user='nlp', password='dongwon', db='chatbot_db', charset='utf8')
    cur = conn.cursor() # 커서생성

    cur.execute("""
                SELECT COUNT(*)
                FROM (
                    SELECT a.message, count(a.message) 
                    FROM (
                        SELECT message
                        FROM fallback_message
                        WHERE fallback_message.fallback_date ='%s'
                    ) AS a
                    GROUP BY a.message
                ) AS b;
                """ %(ymd))
    total = cur.fetchone()[0]

    cur.execute("""
                SELECT a.message, count(a.message) 
                FROM
                (
                    SELECT message
                    FROM fallback_message
                    WHERE fallback_message.fallback_date ='%s'
                ) AS a
                GROUP BY a.message ORDER BY count(a.message) DESC
                LIMIT %s OFFSET %s;
                """ %(ymd, per_page, offset))
    #  "SELECT * FROM dialog ORDER BY intent_no ASC LIMIT %d OFFSET %d;" %(per_page, offset))  # SQL SELECT로 포스트를 가져오되,
    logs = cur.fetchall()
    conn.close()

    # logs = pd.DataFrame((logs))
    # print(len(logs))
    # print(logs)

    # text_count = []
    # if len(logs) > 0:    
    #     for k,v in logs[1].value_counts().items():
    #         text_count.append([k, v])
    return render_template(
        "fallback_temp.html",
        logs = logs,
        ymd = ymd,
        pagination=Pagination(
            page=page,  # 지금 우리가 보여줄 페이지는 1 또는 2, 3, 4, ... 페이지인데,
            total=total,  # 총 몇 개의 포스트인지를 미리 알려주고,
            per_page=per_page,  # 한 페이지당 몇 개의 포스트를 보여줄지 알려주고,
            prev_label="<<",  # 전 페이지와,
            next_label=">>",  # 후 페이지로 가는 링크의 버튼 모양을 알려주고,
            format_total=True,  # 총 몇 개의 포스트 중 몇 개의 포스트를 보여주고있는지 시각화,
        ),
        search=True,  # 페이지 검색 기능을 주고,
        bs_version=5,  # Bootstrap 사용시 이를 활용할 수 있게 버전을 알려줍니다.
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)