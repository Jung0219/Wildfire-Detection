import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# load your CSV
df = pd.read_csv("/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100.csv")

X = df[["det_conf", "temp_conf"]].values
y = df["label"].values

# train logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y)

w = logreg.coef_[0]       # array of [w1, w2]
b = logreg.intercept_[0]  # scalar bias
print("w1 =", w[0], "w2 =", w[1], "b =", b)

# calibrated probabilities
calib_conf = logreg.predict_proba(X)[:, 1]

print("Brier score:", brier_score_loss(y, calib_conf))
print("Log-loss:", log_loss(y, calib_conf))
