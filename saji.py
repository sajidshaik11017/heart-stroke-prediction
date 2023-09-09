#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


# In[2]:


model = pkl.load(open('stroke.pkl','rb'))


# In[3]:


def prediction1(Gender, Age, Hypertension, Heart_Disease,
                                  Ever_Married, Residence_Type, Glucose_Level,
                                  BMI, work_type_Govt_job, work_type_Never_worked,
                                  work_type_Private, work_type_Self_employed, 
                                  work_type_children, smoking_status_Unknown, 
                                  smoking_status_formerly_smoked, smoking_status_never_smoked,
                                  smoking_status_smokes):    
    prediction1 = model.predict([[Gender, Age, Hypertension, Heart_Disease,
                                  Ever_Married, Residence_Type, Glucose_Level,
                                  BMI, work_type_Govt_job, work_type_Never_worked,
                                  work_type_Private, work_type_Self_employed, 
                                  work_type_children, smoking_status_Unknown, 
                                  smoking_status_formerly_smoked, smoking_status_never_smoked,
                                  smoking_status_smokes]])  
    print(prediction1)
    return(prediction1)


# In[4]:


def main():  
      # Now, we will give the title to out web page  
    st.title("Heart Stroke Prediction")  
        
    # Now, we will be defining some of the frontend elements of our web            # page like the colour of background and fonts and font size, the padding and    # the text to be displayed  
    html_temp = """  
    <div style = "background-colour: #FFFF00; padding: 16px">  
    <h1 style = "color: #000000; text-align: centre; "> Streamlit Heart Stroke Prediction ML App   
     </h1>  
    </div>  
    """  
        
    # Now, this line will allow us to display the front-end aspects we have   
    # defined in the earlier  
    st.markdown(html_temp, unsafe_allow_html = True)  
    Gender = st.number_input('Enter your gender ( 1: Male , 0: Female):') 
    Age = st.number_input('Enter your age:')
    Hypertension = st.number_input('Enter hypertension (0 or 1):')
    Heart_Disease = st.number_input('Enter heart disease (0 or 1):')
    Ever_Married = st.number_input('Enter ever married (0 or 1):')
    Residence_Type = st.number_input('Enter residence type (0 or 1):')
    Glucose_Level = st.number_input('Enter glucose level (50 - 300):')
    BMI = st.number_input('Enter bmi (0 - 50):')
    work_type_Govt_job = st.number_input('Government Job (0 or 1):')
    work_type_Never_worked = st.number_input('Never Worked (0 or 1):')
    work_type_Private = st.number_input('Private Job (0 or 1):')
    work_type_Self_employed = st.number_input('Self Employed (0 or 1)')
    work_type_children = st.number_input('Children (0 or 1):')
    smoking_status_Unknown = st.number_input('Smoking Status (0 or 1):')
    smoking_status_formerly_smoked  = st.number_input('Formely Smoked (0 or 1):')
    smoking_status_never_smoked  = st.number_input('Never Smoked (0 or 1):')
    smoking_status_smokes  = st.number_input('Smoked ( 0 or 1):')  
    # Here, the following lines will create the text boxes in which the user can     # enter the data which is required for making the prediction    
    result = " "  
        
    # here, the below line will ensure that whenever the button named 'Predict' # is clicked, the prediction function that is defined earlier is called for making   # the prediction and it will also store it in the variable result  
    if st.button ("Predict"):  
        result = prediction1(Gender, Age, Hypertension, Heart_Disease,
                                  Ever_Married, Residence_Type, Glucose_Level,
                                  BMI, work_type_Govt_job, work_type_Never_worked,
                                  work_type_Private, work_type_Self_employed, 
                                  work_type_children, smoking_status_Unknown, 
                                  smoking_status_formerly_smoked, smoking_status_never_smoked,
                                  smoking_status_smokes)  
    st.success ('The output of the above is {}'.format(result))  
if __name__== '__main__':  
    main()  


# In[ ]:




