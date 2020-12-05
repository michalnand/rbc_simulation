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
dataset = libs_dataset.CellsDataset(training_files, training_labels, testing_files, testing_labels, classes_count = 2, augmentations_count=1)



#train 200 epochs
epoch_count = 200

#cyclic learning rate cheduler
learning_rates  = [0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]

train = libs.Train(dataset, Model0, batch_size = 128, learning_rates = learning_rates)
train.step_epochs(epoch_count, log_path = "../models/model_0")


'''
training result saved into ../models/model_0/result

training progress saved into file training.log columns description
epoch               [int]
training_accuracy   [%]
testing_accuracy    [%]
training_loss_mean  [float]
testing_loss_mean   [float]
training_loss_std   [float]
testing_loss_std    [float]

best model is saved into ../models/model_0/trained
and results into ../models/model_0/result/best.log
'''
