import sys
sys.path.insert(0,'../')

import libs
import libs_dataset
import models.model_0.src.model as Model0


#path where files are stores
path = "/Users/michal/dataset/cells_dataset/sim26/"


#training dats + labels
training_files = []
training_files.append(path + "rbc0_data_sim26.dat")
training_files.append(path + "rbc1_data_sim26.dat")
training_files.append(path + "rbc2_data_sim26.dat")
training_files.append(path + "rbc3_data_sim26.dat")

training_labels = []
training_labels.append(0)
training_labels.append(1)
training_labels.append(0)
training_labels.append(1)


#testing dats + labels
testing_files = []
testing_files.append(path + "rbc0_data_sim26.dat")
testing_files.append(path + "rbc1_data_sim26.dat")
testing_files.append(path + "rbc2_data_sim26.dat")
testing_files.append(path + "rbc3_data_sim26.dat")

testing_labels = []
testing_labels.append(0)
testing_labels.append(1)
testing_labels.append(0)
testing_labels.append(1)

'''
create dataset with pairs training testing
labels corresponds to class IDs
for details see libs_dataset/cells_dataset.py
'''
dataset = libs_dataset.CellsDataset(training_files, training_labels, testing_files, testing_labels, classes_count = 2)
