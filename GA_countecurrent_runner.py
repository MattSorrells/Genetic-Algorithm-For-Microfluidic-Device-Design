import Genetic_Algorithm as GA
import HCA_Multiple_Countercurrent as HCA
import os
# options to pass into genetic algorhtm
options=dict(popSize=100, eliteSize=10, mutationRate=0.05,
             runtime='target_error',generations=500,target_error=0.03)
# solve for device with a given set of constrains using the genetic algorhtm
lengths,df=GA.geneticAlgorithm(**options)
# compute hydraulic circuit analysis for device design on the output from the GA
circiut = HCA.HCA_from_spacing(lengths)
# store results in csvs
filename='Countercurrent Design'
i=0
while os.path.exists(filename+str(i)+'.csv'):
    i+=1
circiut.to_csv(filename+str(i)+'.csv')
