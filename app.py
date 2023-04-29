import streamlit as st 
import numpy as np
import pickle

st.set_option('deprecation.showfileUploaderEncoding',False) 
model = pickle.load(open('finalized_model.sav','rb'))


st.title("The Place of AI in Tackling the Challenge of Malaria in Africa")

st.header("Slide to select malaria case predictors")

st.text('')

col1, col2 = st.columns(2)

with col1:
    c_name = st.slider("Country Name", 0.0, 53.0, 1.0)
    incidence = st.slider("Incidence rate", 0.0, 585.5, 50.0)
    itns = st.slider("% Use ITNs", 0.0, 95.5, 5.0)
    child_fever = st.slider("% Child fever", 0.0, 76.9, 5.0)
    ipt = st.slider("% IPT", 0.0, 59.6, 0.5)
  
   
with col2:
    rural_pop = st.slider("% Rural pop", 0.0, 90.1, 5.0)
    urban_pop = st.slider("% Urban pop", 0.0, 88.9, 5.0)    
    dw_all = st.slider("% Basic DW all", 0.0, 99.9, 5.0)
    san_all = st.slider("% Basic sanitation all", 0.0, 100.0, 5.0)


st.text('')
if st.button("Predict Malaria Case"):
    result = model.predict(np.array([[c_name, incidence, itns, child_fever, ipt, rural_pop, urban_pop, dw_all, san_all]]))
    st.text(result[0])

st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/TeamFlask_notebook)')

# Reference: Santiago Víquez (2023). How to Deploy Machine Learning Models with Python & Streamlit. 
# Accessed on 29/4/2023. https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/