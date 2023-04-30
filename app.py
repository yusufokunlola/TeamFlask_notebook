import streamlit as st 
import numpy as np
import pickle

st.set_option('deprecation.showfileUploaderEncoding',False) 
model = pickle.load(open('finalized_model.sav','rb'))


st.title("")
st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>The Place of AI in Tackling the Challenge of Malaria in Africa</h1>", unsafe_allow_html=True)

st.header("")

st.markdown("<h4 style='text-align: center; color: Black;'>Select variables to predict malaria incidence</h4>", unsafe_allow_html=True)
st.text('')

st.markdown('<p> <strong>Region:</strong> \
            <ul> \
            <li>Central Africa == 0 </li> \
            <li>East Africa == 1</li> \
            <li>North Africa == 2</li> \
            <li>Southern Africa == 3</li> \
            <li>West Africa == 4 </li> \
            </ul> \
        </p>' , unsafe_allow_html=True)

st.text('')
st.text('')
region = st.selectbox(
    'Select Region',
    ('0', '1', '2', '3', '4'))

st.write('You selected:', region)

st.text('')

st.markdown("<h4 style='text-align: center; color: Black;'>Use the slider to select optimal variables</h4>", unsafe_allow_html=True)


st.text('')
col1, col2 = st.columns(2)

with col1:
    rural_pop = st.slider("Rural population (%)", 0.0, 100.0, 5.0)
    itns = st.slider('Use of insecticide-treated bed nets (% of under age 5 population)', 0.0, 100.0, 5.0)
    ipt = st.slider('Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)', 0.0, 100.0, 5.0)
    malaria_case = st.slider("Malaria cases", 0.0, 100.0, 5.0)
  
   
with col2:
    urban_pop = st.slider("Urban population (%)", 0.0, 100.0, 5.0)   
    child_fever = st.slider('Children with fever receiving antimalarial drugs (% of children under age 5 with fever)', 0.0, 100.0, 5.0) 
    dw_all = st.slider("Drinking Water (%)", 0.0, 100.0, 5.0)
    san_all = st.slider("Sanitation (%)", 0.0, 100.0, 5.0)


st.text('')
if st.button("Predict incidence of malaria (per 1,000 population at risk)"):
    result = model.predict(np.array([[region, rural_pop, itns, ipt, malaria_case, urban_pop, child_fever, dw_all, san_all]]))
    st.text(round(result[0],2))

st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/TeamFlask_notebook)')

# Reference: Santiago Víquez (2023). How to Deploy Machine Learning Models with Python & Streamlit. 
# Accessed on 29/4/2023. https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/