import joblib
import numpy as np 
import sklearn
from sentence_transformers import SentenceTransformer
encode_model = SentenceTransformer('all-MiniLM-L6-v2')
path = "C://Users//User//Desktop//git_clone_aml//AppliedMachineLearning//Assignments_aml//"
model_name = "mlp_model.joblib"
model = joblib.load(path + model_name)
def score(text:str, model,threshold:float):
    text = [text]
    features = encode_model.encode(text)
    predicted = model.predict(features)
    propensity = model.predict_proba(features)[:,1]
    return predicted[0], propensity[0]


    

  
