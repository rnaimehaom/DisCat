from flask import Flask, redirect, url_for, render_template, request
import useVAE

app = Flask(__name__)

@app.route("/", methods=['POST', "GET"])
def home():
	return render_template("index.html", molecule_text='', molecule_text1='')

@app.route("/login", methods=["POST", "GET"])
def login():
	if request.method == "POST":
		user = request.form['nm']
		return redirect(url_for('user', usr=user))		
	else:
		return render_template("login.html")

@app.route('/molecule',methods=['POST'])
def molecule():
	val = [float(x) for x in request.form.values()]
	val_1 = val[0]
	molecule = useVAE.main(val_1)

	return render_template('index.html', molecule_text='SMILES Molecule:', molecule_text1=molecule)

@app.route("/<usr>")
def user(usr):
	return f"<h1>{usr}</h1>"

if __name__ == '__main__':
	app.run(debug=True)