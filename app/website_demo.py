from flask import Flask, request, render_template
from ATP_ESM import runner

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        fasta_input = request.form['fasta_input']
        print(fasta_input)
        result = runner.run(fasta_input)
        
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
