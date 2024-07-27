import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Загрузка данных
data = pd.read_csv('titanic_train.csv')

# Предобработка данных
def preprocess_data(data):
    features = data.drop(['Survived'], axis=1)
    labels = data['Survived']
    num_attribs = ['Age', 'Fare']
    cat_attribs = ['Pclass', 'Sex', 'Embarked']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    features_prepared = preprocessor.fit_transform(features)
    return features_prepared, labels

X, y = preprocess_data(data)

# Определение базовых классификаторов
classifiers = [
    ('svc', SVC(probability=True)),
    ('gnb', GaussianNB()),
    ('dt', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier())
]

# Подготовка комбинаций классификаторов
for L in range(1, len(classifiers)+1):
    for subset in combinations(classifiers, L):
        stack = StackingClassifier(estimators=list(subset), final_estimator=LogisticRegression(), cv=5)
        score = cross_val_score(stack, X, y, cv=5, scoring='accuracy').mean()
        print(f'Classifiers {", ".join([name for name, _ in subset])}: Accuracy = {score:.2f}')