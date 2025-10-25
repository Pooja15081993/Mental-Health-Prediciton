import streamlit as st
import pickle 
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *


dt = pickle.load(open('dt_cfs.pkl','rb'))
lr = pickle.load(open('lr_cfs.pkl','rb'))
rf = pickle.load(open('rf_cfs.pkl','rb'))

c1,c2 = st.columns([2,7])

with c1:
    st.image('mental_health.jpg')

with c2:
    st.title('ME_CFS, Depression Prediction Web App')


st.write('Enter the values to generate the prediction')

mname = st.sidebar.selectbox('Select the Model to Train',
                     ['Logistic_Regression','Decision_Tree','Random_Forest'])



col1,col2,col3  = st.columns(3)

with col1:
    age = st.slider('Enter Age',18,70,52)
    sleep_qual = st.number_input('Enter Sleep Qual Index',1.0,10.0,6.2)
    brain_fog = st.number_input('Enter Brain Fog Level',1.0,10.0,5.0)
    phy_pain_scr = st.number_input('Enter Physical Pain Score',1.0,10.0,4.5)
    stress = st.number_input('Enter Stress Level',1.0,10.0,5.4)
   
with col2:   
    dep_phq9 = st.number_input('Enter Depression_phq9 Level',0,27,14)
    fatigue_severity = st.number_input('Enter Fatigue Severity Level',0.0,10.0,4.8)
    pem_hrs = st.slider('Enter PEM Duration Hrs',0,47,18)
    sleep_hrs =  st.slider('Enter Sleep Hrs',3.0,10.0,6.4)
    pem = st.selectbox('PEM Present',['Yes','No'])

with col3:
    gender = st.selectbox('Select Gender',['Male','Female'])
    work_status = st.selectbox('Select Work Status',
                               ['Working','Not Working','Partially Working'])
    social_act = st.selectbox('Select Social Activity Level',
                              ['Very Low','Low','Medium','High','Very High'])
    exr_freq = st.selectbox('Select Exercise frequency',
                                 ['Never','Often','Rarely','Sometimes','Daily'])
    med_mind = st.selectbox('Meditation or Mindfulness',['Yes','No'])
    

# Encoding

if pem == "Yes":
    pem_yes = 1
    pem_no = 0
else:
    pem_yes = 0
    pem_no = 1

if gender == "Male":
    genM = 1
    genF = 0
else:
    genM = 0
    genF = 1

if med_mind == "Yes":
    mmY = 1
    mmN = 0
else:
    mmY = 0
    mmN = 1


if work_status == "Working":
    work = 1
    pwork = 0
    nwork = 0
elif work_status == "Partially Working":
    work = 0
    pwork = 1
    nwork = 0
else:
    work = 0
    pwork = 0
    nwork = 1

if social_act== "Low":
    sa_low = 1
    sa_vl = 0
    sa_m  = 0
    sa_h = 0
    sa_vh = 0
elif social_act== "Very Low":
    sa_low = 0
    sa_vl = 1
    sa_m  = 0
    sa_h = 0
    sa_vh = 0
elif social_act== "Medium":
    sa_low = 0
    sa_vl = 0
    sa_m  = 1
    sa_h = 0
    sa_vh = 0
elif social_act== "Very High":
    sa_low = 0
    sa_vl = 0
    sa_m  = 0
    sa_h = 0
    sa_vh = 1
else:
    sa_low = 0
    sa_vl = 0
    sa_m  = 0
    sa_h = 1
    sa_vh = 0
    
if exr_freq == "Never":
    ef_never = 1
    ef_of = 0
    ef_rar = 0
    ef_st = 0
    ef_d = 0
elif exr_freq == "Often":
    ef_never = 0 
    ef_of = 1
    ef_rar = 0
    ef_st = 0
    ef_d = 0
elif exr_freq == "Sometimes":
    ef_never = 0 
    ef_of = 0
    ef_rar = 0
    ef_st = 1
    ef_d = 0 
elif exr_freq == "Rarely":
    ef_never = 0 
    ef_of = 0
    ef_rar = 1
    ef_st = 0
    ef_d = 0 
else:
    ef_never = 0 
    ef_of = 0
    ef_rar = 0
    ef_st = 0
    ef_d = 1

test_cols  = ['age', 'sleep_quality_index', 'brain_fog_level', 'physical_pain_score',
       'stress_level', 'depression_phq9_score', 'fatigue_severity_scale_score',
       'pem_duration_hours', 'hours_of_sleep_per_night', 'pem_present',
       'gender_Male', 'work_status_Partially working', 'work_status_Working',
       'social_activity_level_Low', 'social_activity_level_Medium',
       'social_activity_level_Very high', 'social_activity_level_Very low',
       'exercise_frequency_Never', 'exercise_frequency_Often',
       'exercise_frequency_Rarely', 'exercise_frequency_Sometimes',
       'meditation_or_mindfulness_Yes']

test_data = [[age,sleep_qual,brain_fog,phy_pain_scr,stress,dep_phq9,fatigue_severity,
              pem_hrs,sleep_hrs,pem_yes,genM,pwork,work,sa_low,sa_m,sa_vh,sa_vl,
              ef_never,ef_of,ef_rar,ef_st,mmY]]

test_df = pd.DataFrame(test_data,columns= test_cols)

st.write('Test Data',test_df)

if mname == "Logistic_Regression":
    pred = lr.predict(test_df)
    pprob = lr.predict_proba(test_df)[0]
elif mname == "Decision_Tree":
    pred = dt.predict(test_df)
    pprob = dt.predict_proba(test_df)[0]
else:
    pred = rf.predict(test_df)
    pprob = rf.predict_proba(test_df)[0]

st.success(pred)

prob_df = pd.DataFrame({
            'Categories': ['Both','Depression','ME_CFS'],
            'Probability': pprob
        }).sort_values('Probability', ascending=False)

st.write(prob_df)
st.bar_chart(prob_df.set_index('Categories'))




