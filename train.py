import sys
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


if __name__ == "__main__":

        df = pd.read_csv("data/data_processed.csv")

        #### Get features ready to model! 
        y = df.pop("cons_general").to_numpy()
        y[y< 4] = 0
        y[y>= 4] = 1

        X = df.to_numpy()
        X = preprocessing.scale(X) # Is standard

        # Impute NaNs
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X)
        X = imp.transform(X)
        penalty = str(sys.argv[1]) if len(sys.argv) > 1 else 'l2'
        C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        solver = str(sys.argv[3]) if len(sys.argv) > 3 else 'lbfgs'
        with mlflow.start_run():

                # Linear model
                clf = LogisticRegression(penalty=penalty, C=C, solver=solver)
                yhat = cross_val_predict(clf, X, y, cv=10)

                acc = np.mean(yhat==y)
                tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)

                mlflow.log_param('penalty', penalty)
                mlflow.log_param('C', C)
                mlflow.log_param('solver', solver)
                mlflow.log_metric('acc', acc)
                mlflow.log_metric('specificity', specificity)
                mlflow.log_metric('sensitivity', sensitivity)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(clf, "model", registered_model_name="LogisticRegressionFarmersModel")
                else:
                        mlflow.sklearn.log_model(clf, "model")


                # Now print to file
                with open("metrics.json", 'w') as outfile:
                        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

                # Let's visualize within several slices of the dataset
                score = yhat == y
                score_int = [int(s) for s in score]
                df['pred_accuracy'] = score_int

                # Bar plot by region

                sns.set_color_codes("dark")
                ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
                ax.set(xlabel="Region", ylabel = "Model accuracy")
                plt.savefig("by_region.png",dpi=80)