# cf_for_wcrf
Counterfactual explanations for Cautious Rando Forest

## Usage of modles

### Load data
```python
data = pd.read_csv("pima.cav")
feature_names = np.array(data.columns[:-1]) # last one id class
feature_names = list(feature_names)
n_features = len(feature_names)
X = np.array(data[feature_names])
Y = np.array(data.iloc[:,-1])

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=None)
```

### Create cautious randon forest
```python
num_trees = 50
s=3
wcrf = WCRF(n_trees=num_trees, s=s, gamma=10, labda=10, tree_max_depth=7, combination=1, data_name='pima')
wcrf.fit(X_train, y_train) # we take equal weights here
# wcrf.fit_w(X_train, y_train)
wcrf_predictions, _, _ = wcrf.predict(X_test)
```

### Create explainer
```python
explainer = RFCFExplainer(wcrf, X_train, feature_names)
```

### Generate counterfactual explanations for indeterminate instance
```python
protected_features = ['Pregnancies','Age']
x = X_test[np.where(wcrf_predictions==-1)[0]]
cf0, cf0_distance = explainer.extract_cf(factuax, objective_class=0, protected_features=protected_features)
cf1, cf1_distance = explainer.extract_cf(factuax, objective_class=1, protected_features=protected_features)
```
