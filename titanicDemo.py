import streamlit as st

# linear algebra
import numpy as np

# data processing
import pandas as pd

#images
from PIL import Image

# data visualization
import seaborn as sns
# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

import plotly.express as px

from bokeh.models.widgets import Div


@st.cache(allow_output_mutation=True)
def loadData():
	train_df = pd.read_csv('train.csv')
	test_df = pd.read_csv('test.csv')
	return train_df,test_df

# Basic preprocessing required for all the models.
@st.cache(suppress_st_warning=True)
def preprocessing():
	train_df = pd.read_csv('train.csv')
	train_df_original =train_df
	test_df = pd.read_csv('test.csv')

	train_df.describe()

	#show me all missing values
	total = train_df.isnull().sum().sort_values(ascending=False)

	#percentage of missing
	percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

	#round off to the nearest one, sort by highest
	percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

	#concatinate the array
	missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
	missing_data.head(5)


	# In[67]:


	survived = 'survived'
	not_survived = 'not survived'

	fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 4))

	women = train_df[train_df['Sex']=='female']
	men = train_df[train_df['Sex']=='male']

	ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
	ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
	ax.legend()
	ax.set_title('Female')

	ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
	ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
	ax.legend()
	ax.set_title('Male')


	# In[68]:


	FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
	FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
	FacetGrid.add_legend()


	# In[69]:


	sns.barplot(x='Pclass', y='Survived', data=train_df)


	# In[70]:


	grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
	grid.map(plt.hist, 'Age', alpha=.5, bins=20)
	grid.add_legend()


	# In[71]:


	data = [train_df, test_df]
	for dataset in data:
	    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
	    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
	    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
	    dataset['not_alone'] = dataset['not_alone'].astype(int)
	train_df['not_alone'].value_counts()


	# In[72]:


	axes = sns.factorplot('relatives','Survived',
	                      data=train_df, aspect = 2.5, )


	# In[73]:


	train_df = train_df.drop(['PassengerId'], axis=1)
	import re
	deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
	data = [train_df, test_df]

	for dataset in data:
	    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
	    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
	    dataset['Deck'] = dataset['Deck'].map(deck)
	    dataset['Deck'] = dataset['Deck'].fillna(0)
	    dataset['Deck'] = dataset['Deck'].astype(int)
	# we can now drop the cabin feature
	train_df = train_df.drop(['Cabin'], axis=1)
	test_df = test_df.drop(['Cabin'], axis=1)


	# In[74]:


	train_df.head()


	# In[75]:


	data = [train_df, test_df]

	for dataset in data:
	    mean = train_df["Age"].mean()
	    std = test_df["Age"].std()
	    is_null = dataset["Age"].isnull().sum()
	    # compute random numbers between the mean, std and is_null
	    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
	    # fill NaN values in Age column with random values generated
	    age_slice = dataset["Age"].copy()
	    age_slice[np.isnan(age_slice)] = rand_age
	    dataset["Age"] = age_slice
	    dataset["Age"] = train_df["Age"].astype(int)
	train_df["Age"].isnull().sum()


	# In[76]:


	train_df['Embarked'].describe()


	# In[77]:


	common_value = 'S'
	data = [train_df, test_df]

	for dataset in data:
	    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


	# In[78]:


	train_df.info()


	# In[79]:


	data = [train_df, test_df]

	for dataset in data:
	    dataset['Fare'] = dataset['Fare'].fillna(0)
	    dataset['Fare'] = dataset['Fare'].astype(int)

	train_df.info()


	# In[80]:


	data = [train_df, test_df]
	titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

	for dataset in data:
	    # extract titles
	    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
	    # replace titles with a more common title or as Rare
	    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
	    # convert titles into numbers
	    dataset['Title'] = dataset['Title'].map(titles)
	    # filling NaN with 0, to get safe
	    dataset['Title'] = dataset['Title'].fillna(0)
	train_df = train_df.drop(['Name'], axis=1)
	test_df = test_df.drop(['Name'], axis=1)


	# In[81]:


	genders = {"male": 0, "female": 1}
	data = [train_df, test_df]

	for dataset in data:
	    dataset['Sex'] = dataset['Sex'].map(genders)


	# In[82]:


	train_df['Ticket'].describe()


	# In[83]:


	train_df = train_df.drop(['Ticket'], axis=1)
	test_df = test_df.drop(['Ticket'], axis=1)


	# In[84]:


	ports = {"S": 0, "C": 1, "Q": 2}
	data = [train_df, test_df]

	for dataset in data:
	    dataset['Embarked'] = dataset['Embarked'].map(ports)


	# In[85]:


	data = [train_df, test_df]
	for dataset in data:
	    dataset['Age'] = dataset['Age'].astype(int)
	    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
	    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
	    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
	    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
	    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
	    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
	    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
	    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

	# let's see how it's distributed
	train_df.head(10)


	# In[86]:


	data = [train_df, test_df]

	for dataset in data:
	    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
	    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
	    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
	    dataset['Fare'] = dataset['Fare'].astype(int)


	train_df.head()


	data = [train_df, test_df]
	for dataset in data:
	    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

	for dataset in data:
	    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
	    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)



	X_train = train_df.drop("Survived", axis=1)
	Y_train = train_df["Survived"]
	X_test  = test_df.drop("PassengerId", axis=1).copy()

	return X_train, X_test, Y_train,train_df_original


# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train):
	# Train the model
	tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
	tree.fit(X_train, y_train)
	y_pred = tree.predict(X_test)
	score = round(tree.score(X_train, y_train) * 100, 2)
	#score = metrics.accuracy_score(y_test, y_pred) * 100
	#report = classification_report(y_test, y_pred)

	return score, tree

# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
	# Scalling the data before feeding it to the Neural Network.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	# Instantiate the Classifier and fit the model.
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score1 = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score1, report, clf

# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred) * 100
	report = classification_report(y_test, y_pred)

	return score, report, clf


# Accepting user data for predicting its Member Type
def accept_user_data():


	class_input = st.selectbox("Select your class: ",
		["First", "Second", "Third"])
	st.write("*The Titanic had 3 Classes. First being the top deck hence closer to the surface.*")

	if class_input == 'First':
		pClass = 1
	elif class_input == 'Second':
		pClass = 2
	else:
		pClass = 3



	sex_input = st.radio("Gender",('Female', 'Male'))
	st.write("*Women & Children first! (famous line from the movie about who was picked to use the limited life boats)* ")


	if sex_input == 'Female':
		sex = 0
	else:
		sex = 1

	titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

	title_input = st.selectbox("Select your title group: ", list(titles.keys()))
	st.write("*Your title determined your social standing hence had an impact to how likely you were to survive*")

	title = titles.get(title_input)

	age_categories = np.array(["0 to 11", "12 to 18", "19 to 22", "23 to 27","28 to 33", "34 to 40", "41 to 66", "67+"])

	age_input = st.selectbox("Select your age group: ", age_categories)
	st.write("*Because children were prioritised for life boats, choose your age group wisely!*")

	if age_input == age_categories[0]:
		age = 0
	elif age_input == age_categories[1]:
		age = 1
	elif age_input == age_categories[2]:
		age = 2
	elif age_input == age_categories[3]:
		age = 3
	elif age_input == age_categories[4]:
		age = 4
	elif age_input == age_categories[5]:
		age = 5
	else:
		age = 6

	sibSp = st.slider('How many siblings / Spouses do you have on board', 0, 8, 1)
	parch = st.slider('How many parents / children do you have on board', 0, 6, 1)
	st.write("*The number of dependents your had on board had an impact on how you planned your escape or who would make an effort to also look for you*")

	fare_input = st.slider('How much was your ticket ?', 7, 250, 150)
	st.write("*The price of your ticket determines your social class (class was important back then)*")

	if fare_input <= 7.91:
		fare = 0
	elif 7.91 < fare_input <= 14.454:
		fare = 1
	elif 14.454 < fare_input <= 31:
		fare = 2
	elif 31 < fare_input <= 99:
		fare = 3
	elif 99 < fare_input <= 250:
		fare = 4
	else:
		fare = 5

	embark_input = st.radio("Where did you embark?",('Cherbourg', 'Queenstown', 'Southampton'))
	st.write("*Take off was in Cherbourg then Queenstown then Southampton theeeeeeen New York*")


	if embark_input == 'Southampton':
		embarked = 0
	elif embark_input == 'Cherbourg':
		embarked = 1
	else:
		embarked = 2

	relatives = sibSp+parch

	if relatives > 0:
		not_alone = 1
	else:
		not_alone = 0

	deck_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

	deck_input = st.radio("What deck were you on?",('A','B','C','D','E','F','G','U'))
	st.write("*couldn't find a proper deck plan, but generally A was a better (safer deck) compared to U*")

	deck = deck_dict.get(deck_input)

	age_class=age*pClass

	fare_per_person = fare_input/relatives
	user_prediction_data = np.array([pClass,sex,age,sibSp,parch,fare,embarked, relatives, not_alone, deck, title, age_class, fare_per_person]).reshape(1,-1)

	return user_prediction_data


def main():
	st.title("How likely were you to survive the titanic")

	image = Image.open('titanic.jpg')
	st.image(image, caption='Sinking titanic',use_column_width=True)
	st.write("The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. \n There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. \n Typical of Hollywood, this disaster was turned into one of the highest grossing films of all time (Titanic) starring Leonardo Decaprio & Kate Winslet with James Cameron directing (Avatar guy). \n ")
	st.write("Depending how you answer the next questions, this model will determine if you would have survived this crash or not?")
	#train_df,test_df = loadData()
	X_train, X_test, y_train,train_df = preprocessing()

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")
		st.write(train_df.head())


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["Decision Tree", "Neural Network", "K-Nearest Neighbours"])

	if(choose_model == "Decision Tree"):
		score, tree = decisionTree(X_train, X_test, y_train)
		st.text("Accuracy of Decision Tree model is: ")
		st.write(score,"%")
		# st.text("Report of Decision Tree model is: ")
		# st.write(report)
		if(st.checkbox("Predict your own",value=True)):

			user_prediction_data = accept_user_data()

			if st.button('Predict!'):


				pred = tree.predict(user_prediction_data)
				prob = tree.predict_proba(user_prediction_data)

				if pred[0] == 0:
					st.write("You would have died😲😲😲😲")
				else:
					st.write("You would have survived!😎😎😎😎")

				st.write("The probability is: ", prob)

	elif(choose_model == "Neural Network"):
		st.title("Coming soon!")
		#score, report, clf = neuralNet(X_train, X_test, y_train)
		#st.text("Accuracy of Neural Network model is: ")
		#st.write(score,"%")
		#st.text("Report of Neural Network model is: ")
		#st.write(report)
		if st.button('Buy us coffee☕️'):
			js = "window.open('https://www.buymeacoffee.com/Sisekelo')"  # New tab or window
			html = '<img src onerror="{}">'.format(js)
			div = Div(text=html)
			st.bokeh_chart(div)
	elif(choose_model == "K-Nearest Neighbours"):
		st.title("Coming soon!")
		#score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
		#st.text("Accuracy of K-Nearest Neighbour model is: ")
		#st.write(score,"%")
		#st.text("Report of K-Nearest Neighbour model is: ")
		#st.write(report)
		if st.button('Buy us coffee☕️'):
			js = "window.open('https://www.buymeacoffee.com/Sisekelo')"  # New tab or window
			html = '<img src onerror="{}">'.format(js)
			div = Div(text=html)
			st.bokeh_chart(div)

	# plt.hist(data['Member type'], bins=5)
	# st.pyplot()

	if st.button('Buy us coffee☕️'):
		js = "window.open('https://www.buymeacoffee.com/Sisekelo')"
		html = '<img src onerror="{}">'.format(js)
		div = Div(text=html)
		st.bokeh_chart(div)

if __name__ == "__main__":
	main()
