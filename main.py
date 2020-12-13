from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

with open('Empattr.pkl','rb') as f:
	model=pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
	return render_template("index.html")


@app.route('/predict',methods=['GET'])
def predict():

	age = request.args.get('age')
	time = request.args.get('time')
	prom = request.args.get('prom')
	rate = request.args.get('rate')
	pay = request.args.get('pay')
	var6 = request.args.get('var6')
	var7 = request.args.get('var7')
	string=[age,time,prom,rate,pay,var6,var7]
	string1=[]
	for i in string:
		d=int(i)
		string1.append(d)
	array=[np.array(string1)]
	prediction=model.predict(array)
	return render_template("index.html",answer=prediction)

if __name__ == "__main__":
	app.run(debug=True)
