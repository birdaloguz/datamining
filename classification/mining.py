import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.svm import LinearSVC

train_set = pd.read_csv("train.csv", index_col="identifier")
test_set = pd.read_csv("test.csv", index_col="identifier")

#plt.scatter(train["condition"], train["highValue"], alpha=0.5)
#plt.title('Scatter plot pythonspot.com')
#plt.xlabel('size')
#plt.ylabel('highValue')

plt.hist(train_set.loc[(train_set['prediction'] == True), 'subway'].astype(int), alpha=0.5, label='True', bins=50)
plt.hist(train_set.loc[(train_set['prediction'] == False), 'subway'].astype(int), alpha=0.5, label='False', bins=50)
plt.legend(loc='upper right')

plt.show()

#Investigate existing company's model
train, test = train_test_split(train_set, test_size=0.2)

y = pd.factorize(train['prediction'])[0]
features = train.columns[:-2]

clf = RandomForestClassifier(n_jobs=2, random_state=0)
model = clf.fit(train[features], y)
test["new_prediction"] = clf.predict(test[features])
precision = precision_score(test["prediction"], test["new_prediction"], average='micro')
print("Random Forest: "+str(precision))


knn = KNeighborsClassifier(n_neighbors=8)
model = knn.fit(train[features], y)
test["new_prediction"] = knn.predict(test[features])
precision = precision_score(test["prediction"], test["new_prediction"], average='micro')
print("KNN: "+str(precision))


nb = GaussianNB()
model = nb.fit(train[features], y)
test["new_prediction"] = model.predict(test[features])
precision = precision_score(test["prediction"], test["new_prediction"], average='micro')
print("Naive Bayes: "+str(precision))


lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train[features], y)
test["new_prediction"] = lr.predict(test[features])
precision = precision_score(test["prediction"], test["new_prediction"], average='micro')
print("Logistic Regression: "+str(precision))

#Accuracy of existing company's model between true predictions
precision = precision_score(train_set.loc[train_set["prediction"]==True]["highValue"], train_set.loc[train_set["prediction"]==True]["prediction"], average='micro')


#Random Forest Prediction of existing company for test set
train, test = train_test_split(train_set, test_size=0.00001)
y = pd.factorize(train['highValue'])[0]
features = train.columns[:-2]

clf = RandomForestClassifier(n_jobs=2, random_state=0)
model = clf.fit(train[features], y)
test_set["highValue"] = clf.predict(test_set[features])


y = pd.factorize(train_set['prediction'])[0]
clf = RandomForestClassifier(n_jobs=2, random_state=0)
model = clf.fit(train_set[features], y)
test_set["prediction"] = clf.predict(test_set[features])


precision = precision_score(test_set.loc[test_set["prediction"]==True]["highValue"], test_set.loc[test_set["prediction"]==True]["prediction"], average='micro')
print(precision)

print("In case of monopoly of existing company...")
#estimated false positives for existing company
ec_fp=len(test_set.loc[(test_set["prediction"]==True) & (test_set["highValue"]==False)])
print("False positives: "+str(ec_fp))
#estimated true positives for existing company
ec_tp=len(test_set.loc[(test_set["prediction"]==True) & (test_set["highValue"]==True)])
print("True positives: "+str(ec_tp))
#budget spent
bs=(ec_fp+ec_tp)*450
print("Budget spent: "+str(bs))
#money earned
me=ec_tp*600+ec_fp*100
print("Money Earned: "+str(me))
#profit
p=me-bs
print("Profit: "+str(p))



#Trials on train.csv to find best classifier to predict high valued properties
train, test = train_test_split(train_set, test_size=0.2)

y = pd.factorize(train['highValue'])[0]
features = train.columns[1:-2]

rfc = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
model = rfc.fit(train[features], y)
test["new_prediction"] = rfc.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("Random Forest: "+str(precision))

clf = MLPClassifier(hidden_layer_sizes=(5,2), max_iter=500, alpha=0.001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(train[features], y)
test["new_prediction"] = clf.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("MLP : "+str(precision))

clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-3)
model = clf.fit(train[features], y)
test["new_prediction"] = clf.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("SGD: "+str(precision))

clf = LinearSVC(C=1.0)
model = clf.fit(train[features], y)
test["new_prediction"] = clf.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("LinearSVC: "+str(precision))

knn = KNeighborsClassifier(n_neighbors=7)
model = knn.fit(train[features], y)
test["new_prediction"] = knn.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("KNN: "+str(precision))

nb = GaussianNB()
model = nb.fit(train[features], y)
test["new_prediction"] = model.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("Naive Bayes: "+str(precision))

lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train[features], y)
test["new_prediction"] = lr.predict(test[features])
precision = precision_score(test["highValue"], test["new_prediction"], average='micro')
print("Logistic Regression: "+str(precision))


###test_set contains the random forest predictions for existing company and high valued propert predictions

#drop the other company's true labeled rows and not high valued properties
contract_set = test_set.loc[(test_set["prediction"]==False)&(test_set["highValue"]==True)]
print(contract_set)


#contract_set["identifier"] = contract_set.index
#contract_set["identifier"].to_csv("identifier.txt", index=False)


print("Our company's profit estimation")
#errors taken into account precision->0.76

#estimated false positives for existing company
ec_fp=len(contract_set)*0.24
print("False positives: "+str(ec_fp))
#estimated true positives for existing company
ec_tp=len(contract_set)*0.76
print("True positives: "+str(ec_tp))
#budget spent
bs=(ec_fp+ec_tp)*450
print("Budget spent: "+str(bs))
#money earned
me=ec_tp*600+ec_fp*100
print("Money Earned: "+str(me))
#profit
p=me-bs
print("Profit: "+str(p))
