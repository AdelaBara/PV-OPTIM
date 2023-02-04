from HR import app
#Requirements: pip install flask, flask_sqlalchemy, flask_wtf, wtforms, WTForms-Alchemy, pymysql, flask_login, flask-bcrypt

if __name__=='__main__':
    app.run(debug=True)
    #app.run(host='192.168.0.111', port='5000', debug=True)