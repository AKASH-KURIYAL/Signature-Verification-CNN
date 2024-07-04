import numpy as np

def multi_output_generator(generator_real,generator_forged):
    while True:
        X_real , y_real_individual = next(generator_real)
        y_real_forged = np.ones(len(y_real_individual))

        X_forged , y_forged_individual = next(generator_forged)
        y_forge_forged = np.zeros(len(y_forged_individual))

        X = np.concatenate((X_real,X_forged),axis=0)
        y_individual = np.concatenate((y_real_individual,y_forged_individual),axis =0)
        y_forged = np.concatenate((y_real_forged,y_forge_forged),axis = 0)

        yield X,{'real_forged':y_forged,'individual':y_individual}