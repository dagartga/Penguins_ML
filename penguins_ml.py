import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

penguin_df = pd.read_csv('../penguins_app/penguins.csv')

penguin_df = penguin_df.dropna()

output = penguin_df['species']
features = penguin_df[['island', 
                       'bill_length_mm', 
                       'bill_depth_mm',
                       'flipper_length_mm',
                       'body_mass_g', 
                       'sex']]

features = pd.get_dummies(features)
# convert species to numeric values
output, uniques = pd.factorize(output) 

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.20)

# instantiate the model
rfc = RandomForestClassifier(random_state=42)
# fit the model
rfc.fit(X_train.values, y_train)
# make predictions on test data
y_pred = rfc.predict(X_test.values)


# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# save the model
with open('random_forest_penguin.pickle', 'wb') as f:
    pickle.dump(rfc, f)
    
# save the species list
with open('output_penguin.pickle', 'wb') as f:
    pickle.dump(uniques, f)
    
# extract and organize the feature importances
importances = rfc.feature_importances_
feature_names = features.columns
feat_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance'])
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

# plot the feature importances
fig, ax = plt.subplots()
ax = sns.barplot(data=feat_importances, x='Importance', y=feat_importances.index, palette='Blues_d')
plt.title('Which features are the most important for prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('rf_feature_importances.png', dpi=200)
