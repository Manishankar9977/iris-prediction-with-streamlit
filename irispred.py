
import time
from numpy import loadtxt 
from keras.models import Sequential 
from keras.layers import Dense 
from numpy import loadtxt 
from numpy import genfromtxt 
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
try:
 from PIL import Image
except ImportError:
 import Image
import streamlit as st
import pandas as pd
import numpy as np
seto=Image.open('set.jpg')
vero=Image.open('ver.jpg')
virg=Image.open('vir.jpg')

iris_data = load_iris()


x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)

data1 = pd.DataFrame(data= np.c_[iris_data['data'], iris_data['target']],columns= iris_data['feature_names'] + ['target'])
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


model = Sequential() 
model.add(Dense(50,input_dim=4,activation='relu')) 
model.add(Dense(25, activation='relu')) 
model.add(Dense(12, activation='relu')) 
model.add(Dense(3, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

model.fit(train_x, train_y, epochs=200, batch_size=10) 
results = model.evaluate(test_x, test_y) 

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

st.title('Predict the Iris Flower:🌻')

st.write("""
### Simple Iris Flower Prediction App
This app predicts the **Iris flower** type! Based on **FEED FORWARD NEURAL NETWORK**
""")

st.sidebar.header('User Input Parameters')

sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

import numpy as np
import pandas as pd
columns=['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)']
def predict(): 
    row = np.array([sepal_length,sepal_width,petal_length,petal_width]) 
    m = pd.DataFrame([row], columns = columns)
    pred = model.predict(m)
    out=pred.flatten()
    if (max(out)==out[0]):
      ou= 'setosa'
      return ou
    elif max(out)==out[1]:
      ou='versicolor'
      return ou
    elif max(out)==out[2]:
      ou='virginica'
      return ou


data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
df=pd.DataFrame(data, index=[0])
st.write(df)

prediction=predict()




col1,col2=st.columns(2)

with col1:
  st.subheader('Sample Dataset')
  st.write(data1)
with col2:
  st.write("""#### Visual description of various features of Iris Species""")
  vv=Image.open('flower.png')
  st.image(vv)
  

  

co1,co2=st.columns(2)

with co1:
  if prediction == 'setosa': 
    st.subheader('Prediction output:-----IRIS SETOSA-----')
    st.image(seto) 
  elif prediction=='versicolor':
    st.subheader('Prediction output:-----IRIS VERSICOLOR-----')
    st.image(vero)
  else :
    st.subheader('Prediction output:-----IRIS VIRGINICA-----')
    st.image(virg)
with co2:
  st.subheader('code')
  code='''
  model = Sequential() 
  model.add(Dense(50,input_dim=4,activation='relu')) 
  model.add(Dense(25, activation='relu')) 
  model.add(Dense(12, activation='relu')) 
  model.add(Dense(3, activation='softmax')) 
  model.compile(loss='categorical_crossentropy', 
  optimizer='adam', metrics=['accuracy'])
  print('Neural Network Model Summary: ')
  print(model.summary())
  model.fit(train_x, train_y, epochs=200, batch_size=10) 
  results = model.evaluate(test_x, test_y) '''
  st.code(code, language='python')



my_bar = st.progress(0)



with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')
