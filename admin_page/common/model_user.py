from flask_login import UserMixin
from common.config import connect_db

class User(UserMixin):
    def __init__(self, user_id, user_name, user_role):
        self.user_id = user_id
        self.user_name = user_name
        self.user_role = user_role
    
    def get_id(self):
        return str(self.user_id)

    def get_name(self):
        return str(self.user_name)
    
    def get_role(self):
        return str(self.user_role)
    
    @staticmethod
    def get_user_info(user_id, user_pw=None):
        result = dict()
        conn, cur = connect_db()
        try:
            sql = ""
            sql += f"SELECT USER_ID, USER_PASSWORD, USER_NAME, PHONE, DEPARTMENT_NAME, USER_ROLE "
            sql += f"FROM admin "
            if user_pw:
                sql += f"WHERE USER_ID = '{user_id}' AND USER_PASSWORD = '{user_pw}'; "
            else:
                sql += f"WHERE USER_ID = '{user_id}'; "
            cur.execute(sql)
            result['data'] = cur.fetchall()[0]
            result['result'] = 'success'
        finally:
            return result

if __name__=='__main__':
    a = User.get_user_info('admin', 'root')
    print(a)
    
        