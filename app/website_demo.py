#from flask import Flask, request, render_template
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_mail import Mail, Message
import json
app = Flask(__name__)
######
app.secret_key = 'abcccc45678'  # for flash message

# 郵件設定
app.config['MAIL_SERVER'] = ''
app.config['MAIL_PORT'] = 0
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = ''
#應用程式密碼
app.config['MAIL_PASSWORD'] = ''

mail = Mail(app)
######
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form['姓名']
    email = request.form['Email']
    message = request.form['訊息']

    msg = Message(subject=f"來自 {name} 的聯絡訊息",
                  sender=email,
                  recipients=['tc930512@gmail.com'],
                  body=f"姓名: {name}\nEmail: {email}\n\n訊息:\n{message}")
    mail.send(msg)
    flash("您的訊息已成功送出，我們會盡快與您聯繫。")
    return redirect(url_for('contactpage'))
######

# 首頁
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

from multiprocessing import Queue, Process


def run_isolated(model_runner, protein_information):
    q = Queue()
    def wrapper(protein_information, q):
        out = model_runner(protein_information)
        q.put(out)
    p = Process(target=wrapper, args=(protein_information, q))
    p.start()
    p.join()
    result = q.get()
    return result

import ATP_Mamba.runner
import ATP_ESM.runner
import ATP_RF_PSSM.runner
import ATP_msCNN.runner
import ATP_RF_BM.runner
import ATP_XGB_PSSM.runner
import ATP_XGB_BM.runner
    
@app.route('/predict', methods=['GET', 'POST'])
def predictpage():
    result = None
    selected_model_value = None
    if request.method == 'POST':
        fasta_input = request.form['fasta_input']
        selected_model_value = request.form['model']
        user_email = request.form.get('user_email', '').strip()

        print(f"Selected model: {selected_model_value}")

        if selected_model_value == "rfpssm":
            result = run_isolated(ATP_RF_PSSM.runner.run_rf_pssm, fasta_input)
        elif selected_model_value == "rfbm":
            result = run_isolated(ATP_RF_BM.runner.run_rf_bm, fasta_input)
        elif selected_model_value == "mscnnpssm":
            result = run_isolated(ATP_msCNN.runner.run_mscnn_pssm, fasta_input)
        elif selected_model_value == "xgbpssm":
            result = run_isolated(ATP_XGB_PSSM.runner.run_xgb_pssm, fasta_input)
        elif selected_model_value == "xgbbm":
            result = run_isolated(ATP_XGB_BM.runner.run_xgb_bm, fasta_input)
        elif selected_model_value == "mamba":
            result = run_isolated(ATP_Mamba.runner.run, fasta_input)
        elif selected_model_value == "esm":
            result = run_isolated(ATP_ESM.runner.run_esm, fasta_input)
        else:
            result = "Invalid model selected or model not handled."

        # ========== Email 寄送 ==========
        if user_email:
            try:
                msg = Message(subject="您的 FASTA 預測結果",
                              sender=app.config['MAIL_USERNAME'],
                              recipients=[user_email],
                              body=f"您好，以下是您所提交的 FASTA 預測結果：\n\nModel: {selected_model_value}\n\n{result}")
                mail.send(msg)
                flash(f'預測結果已成功寄送至 {user_email}！')
            except Exception as e:
                flash(f'Email 傳送失敗：{str(e)}')

    return render_template('predictpage.html', result=result, model=selected_model_value)


#######
import os
import subprocess
from werkzeug.utils import secure_filename

@app.route('/weblogo', methods=['GET', 'POST'])
def weblogo():
    logo_path = None
    if request.method == 'POST':
        os.makedirs('static', exist_ok=True)

        file = request.files['fasta_file']
        user_email = request.form.get('user_email', '')

        if not file or file.filename == '' or not user_email:
            flash('請上傳檔案並填寫 Email')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = os.path.join('static', filename)
        file.save(input_path)

        output_path = os.path.join('static', 'logo.png')

        try:
            subprocess.run([
                'weblogo',
                '--format', 'png',
                '--resolution', '300',
                '--errorbars', 'NO',
                '--color-scheme', 'chemistry',  # 適合蛋白質的顏色方案
                '--sequence-type', 'protein',
                '--fin', input_path,
                '--fout', output_path
            ], check=True)
            logo_path = 'logo.png'

            # 寄信
            msg = Message(subject="您的 WebLogo 圖片",
                          sender=app.config['MAIL_USERNAME'],
                          recipients=[user_email],
                          body="您好，以下是您剛上傳的 FASTA 序列所產生的 WebLogo 圖片。\n\n感謝使用！")

            with app.open_resource(output_path) as fp:
                msg.attach("weblogo.png", "image/png", fp.read())

            mail.send(msg)
            flash(f'圖檔已成功寄送至 {user_email}！')

        except subprocess.CalledProcessError as e:
            flash('WebLogo 產生失敗，請確認輸入格式正確')
        except Exception as e:
            flash(f'Email 傳送失敗：{str(e)}')

    return render_template('weblogo.html', logo_path=logo_path)

def setup_email_config():
    global app
    # 郵件設定
    with open('secret.json') as f:
        email_config = json.load(f)
    app.config['MAIL_SERVER'] = email_config['mail_server']
    app.config['MAIL_PORT'] = email_config['mail_port']
    app.config['MAIL_USE_TLS'] = email_config['mail_use_tls']
    app.config['MAIL_USERNAME'] = email_config['mail_username']
    #應用程式密碼
    app.config['MAIL_PASSWORD'] = email_config['mail_password']


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
