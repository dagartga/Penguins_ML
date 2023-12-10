import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Penguin Species Classifier')
st.write('This app uses 6 inputs to predict the species of a penguin using a model' 
         ' built on the Palmer Penguins dataset. Use the form below to get started!')

# setup a password to protect the app
password_guess = st.text_input('Please Enter the Password:')
if password_guess != st.secrets['password']:
    st.stop()


rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')

# load the pre-trained model
rfc = pickle.load(rf_pickle)
# load the species map
unique_penguin_map = pickle.load(map_pickle)

rf_pickle.close()
map_pickle.close()

# code for all of the user inputs
with st.form("user_inputs"):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()

# format the user inputs for the model
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgersen':
    island_torgerson = 1
    
sex_male, sex_female = 0, 0
if sex == 'Male':
    sex_male = 1
elif sex == 'Female':
    sex_female = 1
    
# make the prediction
new_prediction = rfc.predict([[bill_length, 
                               bill_depth, 
                               flipper_length, 
                               body_mass, 
                               island_biscoe,
                               island_dream,
                               island_torgerson,
                               sex_female,
                               sex_male]])

# map the prediction to the species name
prediction_species = unique_penguin_map[new_prediction][0]
    
# display the prediction
st.subheader(f'The predicted species for your penguin is {prediction_species}')

# display the feature importances
st.header('Feature Importances')
st.write(
    """
         We used a machine learning model (Random Forest)
         to predict the species, the features used in this prediction
         are ranked by relative importance below.
"""
)

st.image('rf_feature_importances.png')


# load the data
penguin_df = pd.read_csv('penguins.csv')
penguin_df = penguin_df.dropna()

# display the distribution of the species for bill length
fig, ax = plt.subplots()
ax = sns.histplot(data=penguin_df, x='bill_length_mm', hue='species', palette='colorblind')
plt.title('Distribution of Bill Length by Species')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Count')
# plot the user input value
plt.axvline(x=bill_length, color='black', linestyle='--')
plt.tight_layout()
st.pyplot(fig)