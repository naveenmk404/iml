import numpy as np
import pandas as pd

data = pd.read_csv('enjoysport.csv')

concepts = np.array(data.iloc[:,:-1])

target = np.array(data.iloc[:,-1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print(specific_h)
    general_h = [['?' for i in range (len(specific_h))] for i in range (len(specific_h))]
    print(general_h[0])

    for i, h in enumerate(concepts):
        print(f"Instace {i+1} is {h}")
        if target[i]=='yes':
            print("Instance {i+1} is positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x]='?'
                    specific_h[x]='?'
        
        elif target[i]=='no':
            print("Instance is negative")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else :
                    general_h[x][x]='?'
        
        print(f"After instance {i+1} general hypothesis is : {general_h} ")
        print(f"After instance {i+1} specific hypothesis is : {specific_h} ")
        
        
    indices = [i for i, val in enumerate(general_h) if val==['?', '?', '?', '?', '?', '?']]
    
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    return general_h, specific_h

general_h, specific_h = learn(concepts,target)

print('--------------------------------------------------------------------------------')
print("General Hypothesis : ",general_h)
print("Specific Hypothesis : ", specific_h)