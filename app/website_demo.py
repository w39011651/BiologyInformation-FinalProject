from flask import Flask, request, render_template
from ATP_Mamba import runner
from multiprocessing import Queue, Process

app = Flask(__name__)

def run_in_process(protein_information):
    q = Queue()
    def wrapper(protein_information, q):
        out = runner.run(protein_information)
        q.put(out)
    p = Process(target=wrapper, args=(protein_information, q))
    p.start()
    p.join()
    result = q.get()
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        fasta_input = request.form['fasta_input']
        print(fasta_input)
        result = run_in_process(fasta_input)
        
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)