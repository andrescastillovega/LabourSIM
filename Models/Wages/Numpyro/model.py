

class  Model():
    def __init__(self, name, parameters):
        self.name = name
        if not any(key in parameters for key in ["intercept", "slope"]):
            raise ValueError("parameters dictionary must contain keys 'intercept' and 'slope'")
        self.parameters = parameters

    def __str__(self):
        return self.name
    
    def standardize_var(self, variable):
        variable = (variable - variable.mean()) / variable.std()
        return variable
    
    def data_processing(self, dataset, standardize_vars=[]):

    
    def build(self, dataset, target_var="salary", standardize_vars=[], year=None, parameterization="centered"):
        self.dataset = dataset.to_records(index=True)

        
        