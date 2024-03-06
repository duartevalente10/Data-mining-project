from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import joblib 

# importar os dados
data = pd.read_csv('C:\\Users\\duart\\Desktop\\AMD\\FP_A\\Data\\exportview2.csv')

# separar as colunas dos atributos da coluna das lentes 
X = data[['eyeage', 'prescription', 'astigmatic', 'tear_rate']]
y = data['lenses']

#  fazer one-hot encoding dos dados pois o classificador não aceita dados não numericos como entrada
X = pd.get_dummies(X, drop_first=False)

# dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# criar o modelo de classificação Naive Bayes
model = CategoricalNB()

# treinar o classificador com os dados de treino
model.fit(X_train, y_train)

# guardar o modelo
joblib.dump(model, 'Models/naive_bayes_model.pkl')

#-------------------------------------------------#
#-----------------PREDICTIONS---------------------#
#-------------------------------------------------#

# importar o modelo
loaded_model = joblib.load('Models/naive_bayes_model.pkl')

# fazer as previsoes para os dados de teste
y_pred = loaded_model.predict(X_test)

# calcular a accuracy, precision, recall e F1-score
report = classification_report(y_test, y_pred, target_names=loaded_model.classes_)

print(report)