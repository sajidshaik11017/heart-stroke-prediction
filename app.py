# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:55:27 2023

@author: sajid
"""

import numpy as np
import pickle
import streamlit 

loaded_model = pickle.load(open('C:/Users/sajid/ANISA/Machine learning projects\heart stroke prediction/trained_model.sav','rb'))