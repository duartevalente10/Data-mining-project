import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# importar os dados
#data = pd.read_csv('C:\\Users\\duart\\Desktop\\AMD\\FP_A\\Data\\exportview2.csv')

# nomes dos atributos
#attribute_columns = ['eyeage', 'prescription', 'astigmatic', 'tear_rate']

# separar as colunas dos atributos da coluna das lentes 
#X = data[attribute_columns]
#y = data['lenses']

# importar pos dados
data2 = pd.read_csv('C:\\Users\\duart\\Desktop\\AMD\\_AMD_finalProject_A1\\dataset_long_name_ORIGINAL.tab', delimiter='\t',skiprows=[1,2])

attribute_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# importar pos dados do tab
# selecionar as colunas das fetures
X = data2[attribute_columns]
y = data2['class']   # selecionar a coluna da classe

# dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vars para armazenar regras e erros
attribute_rules = {}
attribute_errors = {}
attribute_total_errors = {}

# ciclo para precorrer cada atributo
for attribute in attribute_columns:
    attribute_rules[attribute] = {}
    # para cada tipo de valor do atributo
    for value in X_train[attribute].unique():
        # indexar ás regras o valor do atributo 
        attribute_rules[attribute][value] = {}
        # para cada tipo de lente
        for class_value in y_train.unique():
            # contar a frequência de cada valor dos atributo para cada tipo de lente
            count = len(X_train[(X_train[attribute] == value) & (y_train == class_value)])
            attribute_rules[attribute][value][class_value] = count

        # calcular o erro para cada valor de atributo
        error = sum(attribute_rules[attribute][value].values()) - max(attribute_rules[attribute][value].values())
        attribute_errors[attribute] = error

# escolher os pares com menor erro
min_error = min(attribute_errors.values())

# nomes dos atributos com menor erro
best_attributes = [attribute for attribute, error in attribute_errors.items() if error == min_error]

# dos melhores atributos selecionar um aleatoriamente
selected_attribute = random.choice(best_attributes)

print("Atributo selecionado:", selected_attribute)

#print("Valores possiveis do atributo: ", X_train[selected_attribute].unique())

print("Numero de instancias do atributo: ", len(X_train[selected_attribute]))

#print("Classe corespondente para cada instancia: ", attribute_rules[selected_attribute])

print("Regras para cada valor do atributo:")

# Open the file in write mode
with open("Outputs/oneR_OUTPUT.txt", "w") as output_file:
    # para cada possivel valor do atributo selecionado
    for value in X_train[selected_attribute].unique():
        # numero de cada occorencias de cada classe
        class_counts = attribute_rules[selected_attribute][value]
        # escolher o valor da classe com maior occorencias
        predicted_class = max(class_counts, key=class_counts.get)
        # calcular o erro 
        error = sum(attribute_rules[selected_attribute][value].values()) - max(attribute_rules[selected_attribute][value].values())
        # numero total de instancias do valor
        total = sum(attribute_rules[selected_attribute][value].values())
        # guardar no ficheiro os valores
        output_file.write(f"({selected_attribute}, {value}, {predicted_class}) : ({error}, {total})\n")
        print(f"({selected_attribute}, {value}, {predicted_class}) : ({error}, {total})")

# guardar a regra 
with open("Models/regra.txt", "w") as arquivo:
    arquivo.write(selected_attribute)

#-------------------------------------------------#
#-----------------PREDICTIONS---------------------#
#-------------------------------------------------#

# importar a regra para as previsoes
with open("Models/regra.txt", "r") as arquivo:
    regraImportada = arquivo.read()

#print(regraImportada)

# var para guardar a previsao de cada tipo de lente
predictions = {}

# ciclo para precorrer cada linha dos dados te teste
for index, row in X_test.iterrows():
    # verificar o valor do melhor atributo encontrado anteriormente
    selected_value = row[regraImportada]

    # encontar o tipo de lente com a maior ocorrencia para o valor do atributo selecionado
    class_counts = attribute_rules[regraImportada][selected_value]
    predicted_class = max(class_counts, key=class_counts.get)

    # guardar o valor encontrado com maior probabilidade
    predictions[index] = predicted_class

# converter para o tipo "Series" do pandas
y_pred = pd.Series(predictions, name='predicted_class')

# calcular a accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
