import csv

a = []

with open ('enjoysport.csv','r') as csvfile:
    next(csvfile)
    for row in csv.reader(csvfile):
        a.append(row)

num_attributes = len(a[0])-1

hypothesis = ['0']*num_attributes

for i in range(0,len(a)):
    if(a[i][num_attributes]=='yes'):
        print(f'instance {i+1} is postivie')
        for j in range(0,num_attributes):
            if hypothesis[j]=='0' or hypothesis[j]==a[i][j] :
                hypothesis[j]=a[i][j]
            else :
                hypothesis[j]='?'
        print(f"Hypothesis H{i} is {hypothesis}")
    
    else :
        print(f"instance {i+1} is negative and ignored")
        print(f"Hypothesis H{i} is {hypothesis}")
        
        
print(f'Most specific hypothesis : {hypothesis}')