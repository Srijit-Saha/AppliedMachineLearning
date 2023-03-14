import score 
import joblib
from flask import Flask, request, render_template, url_for, redirect

app = Flask(__name__)

model_path = "C://Users//User//Desktop//git_clone_aml//AppliedMachineLearning//Assignments_aml//mlp_model.joblib"
model=joblib.load(model_path)
threshold=0.7


@app.route('/') 
def home():
    return render_template('spam_page.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop = score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans = f"""The sentence "{sent}" is {lbl} with propensity {prop}."""
    return render_template('result_page.html', ans=ans)


if __name__ == '__main__': 
    app.run(debug=True)