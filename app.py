import streamlit as st
import pandas as pd
from PIL import Image
from pycaret.classification import *
data=pd.read_csv('final_data_clean.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
model = load_model('Movie Success Prediction')




Geners=data.columns[4:4+20]    # Geners
Actors=data.columns[24:24+70] # Actors
Directors=data.columns[94:94+30] # Directors
img_title= Image.open('img_title.jpg')
img_flop= Image.open('flop.jfif')
img_hit= Image.open('hits-movies.png')
st.title("  Movies Success Prediction")
st.image(img_title)
st.sidebar.title("Features")
Runtime=st.sidebar.slider('Runtime  ( Minutes)',66,175,160)
Votes=st.sidebar.slider('Votes',1,100000,5000)
Revenue=st.sidebar.slider('Revenue (Millions)',0,220,120)
Rating=st.sidebar.slider('Rating',0,10,5)
Geners1=st.sidebar.selectbox('select Gener first ',Geners.tolist())
Geners2=st.sidebar.selectbox('select Gener Second ',Geners.tolist())
Geners3=st.sidebar.selectbox('select Gener Third ',Geners.tolist())
Actors1=st.sidebar.selectbox('select first Actor ',Actors.tolist())
Actors2=st.sidebar.selectbox('select Second Actor ',Actors.tolist())
Actors3=st.sidebar.selectbox('select Third Actor ',Actors.tolist())
Director=st.sidebar.selectbox('select Director',Directors.tolist())
Geners_list=list(set([Geners1,Geners2,Geners3]))
Actors_list=list(set([Actors1,Actors2,Actors3]))
Director_list=list(Director)
selected_columns=Geners_list+Actors_list+Director_list
data_columns=data.columns[4:-1].to_list()
dummy_list=[Runtime,Rating,Votes,Revenue]
for i in range(len(data_columns)):
    if data_columns[i] in selected_columns:
        dummy_list.append(1)
    else:
        dummy_list.append(0)
#dummy_list
data_unseen=pd.DataFrame(dummy_list).T
data_unseen.columns=data.columns[:-1]
label=predict_model(model,data_unseen)['Label'][0]
if label==1:
	st.image(img_hit)
else:
	st.image(img_flop)
