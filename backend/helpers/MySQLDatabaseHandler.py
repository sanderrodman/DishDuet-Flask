import os
import sqlalchemy as db

class MySQLDatabaseHandler(object):
    
    # Check for Cornell deployment server signature
    IS_CORNELL_DOCKER = True if 'DB_NAME' in os.environ else False
    # Check for general deployment server signature (like Railway)
    IS_GENERAL_DEPLOYMENT = True if 'MYSQLDATABASE' in os.environ else False

    def __init__(self,MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE,MYSQL_HOST = "localhost"):
        
        if self.IS_GENERAL_DEPLOYMENT:
            self.MYSQL_HOST = os.environ.get('MYSQLHOST', MYSQL_HOST) or MYSQL_HOST
            self.MYSQL_USER = os.environ.get('MYSQLUSER', MYSQL_USER) or MYSQL_USER
            self.MYSQL_USER_PASSWORD = os.environ.get('MYSQLPASSWORD', '')
            self.MYSQL_PORT = int(os.environ.get('MYSQLPORT', MYSQL_PORT))
            self.MYSQL_DATABASE = os.environ.get('MYSQLDATABASE', MYSQL_DATABASE) or MYSQL_DATABASE
        elif self.IS_CORNELL_DOCKER:
            self.MYSQL_HOST = os.environ.get('DB_NAME', MYSQL_HOST)
            self.MYSQL_USER = "admin"
            self.MYSQL_USER_PASSWORD = "admin"
            self.MYSQL_PORT = 3306
            self.MYSQL_DATABASE = "kardashiandb"
        else:
            self.MYSQL_HOST = MYSQL_HOST
            self.MYSQL_USER = MYSQL_USER
            self.MYSQL_USER_PASSWORD = MYSQL_USER_PASSWORD
            self.MYSQL_PORT = MYSQL_PORT
            self.MYSQL_DATABASE = MYSQL_DATABASE

        self.engine = self.validate_connection()

    def validate_connection(self):
        print(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")
        return db.create_engine(f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_USER_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}")

    def lease_connection(self):
        return self.engine.connect()
    
    def query_executor(self,query):
        conn = self.lease_connection()
        if type(query) == list:
            for i in query:
                conn.execute(i)
        else:
            conn.execute(query)
        
    def query_selector(self,query):
        conn = self.lease_connection()
        data = conn.execute(query)
        return data

    def load_file_into_db(self,file_path  = None):
        if self.IS_CORNELL_DOCKER or self.IS_GENERAL_DEPLOYMENT:
            return
        if file_path is None:
            file_path = os.path.join(os.environ['ROOT_PATH'],'init.sql')
        sql_file = open(file_path,"r")
        sql_file_data = list(filter(lambda x:x != '',sql_file.read().split(";\n")))
        self.query_executor(sql_file_data)
        sql_file.close()

