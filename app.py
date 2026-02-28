from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/")
def dashboard():
    # Load the predictions generated in Jupyter
    df = pd.read_csv("predicted_next_month.csv")
    
    # Send only the top 100 most urgent points to the UI for performance
    data = df.head(100).to_dict(orient="records")

    return render_template("dashboard.html",
                           data=data,
                           total=len(df),
                           high=len(df[df['Risk_Level'] == 'High']),
                           medium=len(df[df['Risk_Level'] == 'Medium']),
                           low=len(df[df['Risk_Level'] == 'Low']))

if __name__ == "__main__":
    app.run(debug=True)