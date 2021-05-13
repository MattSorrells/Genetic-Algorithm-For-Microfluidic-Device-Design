import Genetic_Algorithm as GA
import HCA_Multiple_Countercurrent as HCA
import os

options=dict(popSize=100, eliteSize=10, mutationRate=0.05,
             runtime='target_error',generations=500,target_error=0.03)
lengths,df=GA.geneticAlgorithm(**options)
circiut = HCA.HCA_from_spacing(lengths)

filename='Countercurrent Design'
i=0
while os.path.exists(filename+str(i)+'.csv'):
    i+=1
circiut.to_csv(filename+str(i)+'.csv')
