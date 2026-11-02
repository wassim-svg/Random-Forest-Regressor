import pandas as pd
import seaborn as sns

healthexp = sns.load_dataset('healthexp')
healthexp.head(10)

healthexp = pd.get_dummies(healthexp)
X = healthexp.drop(['Categorie'], axis= 1)
y = healthexp['Categorie']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y, random_state=19, text_size= 0.2)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=13)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_squared_error(y_pred, y_test)
mean_squared_error(y_pred, y_test)
r2_score(y_pred,y_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 20 , 30],
    'min_samples_split': [2, 5 ,10],
    'min_samples_leaf': [1, 2, 4]
}

from sklearn.model_selection import GridSearchCV

rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_me')

rf_cv.fit(X_train, y_train)
y_pred = rf_cv.predict(X_test)
mean_squared_error(y_pred, y_test)
mean_squared_error(y_pred, y_test)
r2_score(y_pred,y_test)