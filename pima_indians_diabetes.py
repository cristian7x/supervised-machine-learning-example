# Author: Cristian Camilo Perez
# Email: cristian.camilo.2594@gmail.com
# GitHub: https://github.com/cristian7x

from pandas import read_csv
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
import matplotlib.pyplot as plt


# Calculate metrics and measures
def calculate_metrics(n_samples, y_test, y_score):
	conf_m = confusion_matrix(y_test, y_score)
	tn, fp, fn, tp = conf_m.ravel()
	accuracy = accuracy_score(y_test, y_score)*100
	error = ((fn+fp)/n_samples)*100
	tpr = tp/(tp+fn) # sensitivity
	fnr = fn/(tp+fn)
	tnr = tn/(tn+fp) # specificity
	fpr = fp/(tn+fp)
	ppv = tp/(tp+fp) # precision
	macroaverage = (tpr+tnr)/2
	return n_samples, (y_test != y_score).sum(), conf_m, accuracy, error, tpr, fnr, tnr, fpr, ppv, macroaverage

# Print metrics
def print_metrics(metrics):
	print("Condition Negative Rate: {1}/{0}".format(*metrics))
	print("Confusion Matrix:\n{2}".format(*metrics))
	print("Accuracy: {3:.3f}%".format(*metrics))
	print("Error: {4:.3f}%".format(*metrics))
	print("TPR (Sensitivity): {5:.3f}".format(*metrics))
	print("FNR: {6:.3f}".format(*metrics))
	print("TNR (Specificity): {7:.3f}".format(*metrics))
	print("FPR: {8:.3f}".format(*metrics))
	print("PPV (Precision): {9:.3f}".format(*metrics))
	print("Macroaverage: {10:.3f}".format(*metrics))
	print("*******************************************")

# Save decision tree in txt. For visualization open http://webgraphviz.com/
def save_tree_txt(tree, name):	
	with open(name, "w") as f:
		f = export_graphviz(tree, out_file=f, feature_names=['pregnancies', 'glucose', 'bloodpresure', 'skinthickness', 'insulin', 'bmi', 'dpf', 'age'])


# Data Import
diabetes_df = read_csv('diabetes.csv', sep=',', header=None)

# Data preprocessing
# Z-Score Normalization
for each in diabetes_df.columns[:-1]:
  mean, std = diabetes_df[each].mean(), diabetes_df[each].std()
  diabetes_df.loc[:, each] = (diabetes_df[each] - mean)/std

# Data Slicing
X = diabetes_df.values[:, 0:8]
y = diabetes_df.values[:, 8]

# Create training and testing sets
# Training Set 77%
# Testing Set 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
n_samples = X_test.shape[0]


# Classifiers
# Decision Tree - Criterion: Entropy
decision_tree = DecisionTreeClassifier(criterion="entropy")
decision_tree.fit(X_train, y_train)
y_score_dt = decision_tree.predict(X_test)
save_tree_txt(decision_tree, "decision_tree.txt")

# Gaussian Naive-Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_score_nb = gnb.predict(X_test)

# Random Forest - trees: 73
rf_clf = RandomForestClassifier(n_estimators=73)
rf_clf.fit(X_train, y_train)
y_score_rf = rf_clf.predict(X_test)

# SVM
C = 1.0
svc = SVC(kernel="linear", C=C)
svc.fit(X_train, y_train)
y_score_svc = svc.predict(X_test)

# Calculating metrics and measures
metrics_dt_gini = calculate_metrics(n_samples, y_test, y_score_dt)
metrics_nb = calculate_metrics(n_samples, y_test, y_score_nb)
metrics_rf = calculate_metrics(n_samples, y_test, y_score_rf)
metrics_svc = calculate_metrics(n_samples, y_test, y_score_svc)
print("Decision Tree")
print_metrics(metrics_dt_gini)
print("Naive Bayes")
print_metrics(metrics_nb)
print("Random Forest")
print_metrics(metrics_rf)
print("SVM")
print_metrics(metrics_svc)

# Cross-Validation kFOLD:10

scores_dt = cross_val_score(decision_tree, X, y, cv=10)
scores_nb = cross_val_score(gnb, X, y, cv=10)
scores_rf = cross_val_score(rf_clf, X, y, cv=10)
scores_svc = cross_val_score(svc, X, y, cv=10)
print("Accuracy KFold:10")
print("Decision Tree {0:0.2f}% (+/- {1:0.2f})".format(scores_dt.mean()*100, scores_dt.std()))
print("Naive Bayes {0:0.2f}% (+/- {1:0.2f})".format(scores_nb.mean()*100, scores_nb.std()))
print("Random Forest {0:0.2f}% (+/- {1:0.2f})".format(scores_rf.mean()*100, scores_rf.std()))
print("SVM {0:0.2f}% (+/- {1:0.2f})".format(scores_svc.mean()*100, scores_svc.std()))

# ROC Curve
# Compute ROC curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_score_dt)
auc_dt = auc(fpr_dt, tpr_dt)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_score_nb)
auc_nb = auc(fpr_nb, tpr_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
auc_rf = auc(fpr_rf, tpr_rf)
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_score_svc)
auc_svc = auc(fpr_svc, tpr_svc)

# Draw Curve
plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color='darkorange',
         lw=lw, label="Decision Tree (area = {0:.2f})".format(auc_dt))
plt.plot(fpr_nb, tpr_nb, color='purple',
         lw=lw, label="Naive Bayes (area = {0:.2f})".format(auc_nb))
plt.plot(fpr_rf, tpr_rf, color='teal',
         lw=lw, label="Random Forest (area = {0:.2f})".format(auc_rf))
plt.plot(fpr_svc, tpr_svc, color='crimson',
         lw=lw, label="SVM (area = {0:.2f})".format(auc_svc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
