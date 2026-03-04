from flask import Flask, render_template, request
import numpy as np
from scipy.stats import t

app = Flask(__name__)

def ttest(data, mu0, alpha=0.05):

    data = np.array(data)
    n = len(data)

    xbar = np.mean(data)
    s = np.std(data, ddof=1)

    se = s / np.sqrt(n)

    t_cal = (xbar - mu0) / se
    df = n - 1

    t_crit = t.ppf(1 - alpha/2, df)
    p_value = 2 * (1 - t.cdf(abs(t_cal), df))

    reject = abs(t_cal) > t_crit

    return {
        "mean": round(xbar,4),
        "std_dev": round(s,4),
        "t_score": round(t_cal,4),
        "p_value": round(p_value,4),
        "decision": "Reject H0" if reject else "Fail to Reject H0"
    }

@app.route("/", methods=["GET","POST"])
def home():

    result = None

    if request.method == "POST":

        data = request.form["data"]
        mu0 = float(request.form["mu0"])

        data = list(map(float, data.split(",")))

        result = ttest(data, mu0)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)