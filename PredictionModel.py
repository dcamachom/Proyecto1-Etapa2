from joblib import load, dump

class Model:

    def __init__(self,columns):
        self.model = load("assets/model.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
    
    def retrain_model(self, new_data):
        X = new_data.drop(columns=["Class"]) 
        y = new_data["Class"]  
        
        self.model.fit(X, y)
        
        dump(self.model, "assets/modelo_reentrenado.joblib")
