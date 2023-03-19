# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:45:24 2023

@author: sajid
"""

import numpy as np
import pickle


loaded_model = pickle.load(open('C:/Users/sajid/ANISA/Machine learning projects\heart stroke prediction/trained_model.sav','rb'))

input_data = (1,67.0,0 ,1 ,1 ,2 ,1 ,228.69 ,36.6,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] ==0):
    print('The person is not heart stroke')
else:
    print('The person is heart stroke')