import numpy
import torch
from .dats_load import *

class CellsDataset:
    
    '''
    create dataset for classification
    usage :

    1, load data, just calling constructor
        @param training_files   : list of paths to dats files
        @param training_labels  : list of class IDs, integer numbers, from range <0, classes_count)
        @param testing_files    : list of paths to dats files 
        @param testing_labels   : list of class IDs, integer numbers, from range <0, classes_count)

        @param classes_count    : number of classes
        @param window_size      : time sequence window size
        @param cols             : list which colums will be readed from dats files

        @param augmentations_count : count of differen augmentations for training data
    
    2, obtain input x, and target output by calling :
        x, y = dataset.get_training_batch()

        x.shape = (batch_size, len(cols), window_size)
        y.shape = (batch_size, classes_count)

        note : for classes y, one-hot encoding is used

    note : real dataset is too big to hold in RAM (I have only 32G)
    that's why dataset is created runtime
    '''
    def __init__(self, training_files, training_labels, testing_files, testing_labels, classes_count, window_size = 1024, cols = [1, 2, 3], augmentations_count = 32):

        self.width         = window_size
        self.channels      = len(cols)
        self.input_shape   = (self.channels, self.width)

        self.classes_count = classes_count
        self.output_shape  = (self.classes_count, )

        self.augmentations_count = augmentations_count
        
        self.training_dats      = DatsLoad(training_files, cols = cols)
        self.training_labels    = training_labels

        self.testing_dats       = DatsLoad(testing_files, cols = cols)
        self.testing_labels     = testing_labels

        self.training_count = (1 + self.augmentations_count)*self.training_dats.data.shape[0]*self.training_dats.data.shape[1]
        self.testing_count  = self.testing_dats.data.shape[0]*self.testing_dats.data.shape[1]

       
        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_dats shape    = ", self.training_dats.data.shape)
        print("testing_dats shape    = ", self.testing_dats.data.shape)
        print("training_count  = ", self.get_training_count())
        print("testing_count   = ", self.get_testing_count())
        print("channels_count  = ", self.channels)
        print("sequence_length = ", self.width)
        print("classes_count   =  ", self.classes_count)

        x, y = self.get_training_batch(batch_size=32)
        print("batch(32) tensor shape = ", x.shape, y.shape)
        print("\n\n\n\n") 

    def get_training_count(self):
        return self.training_count

    def get_testing_count(self):
        return self.testing_count

    def get_training_batch(self, batch_size = 128):
        return self._get_batch(self.training_dats.data, self.training_labels, batch_size, agumentation = True)

    def get_testing_batch(self, batch_size = 128):
        return self._get_batch(self.testing_dats.data, self.testing_labels, batch_size)


    def _get_batch(self, x, y, batch_size = 128, agumentation = False):
        cells_count = x.shape[0]
        time_steps  = x.shape[1]

        result_x = torch.zeros((batch_size, self.channels, self.width))
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size): 
            cell_idx = numpy.random.randint(cells_count)
            time_idx = numpy.random.randint(time_steps - self.width)

            tmp = x[cell_idx][time_idx:time_idx + self.width]
            tmp = tmp.transpose()

            class_id = y[cell_idx]
            result_x[i]  = torch.from_numpy(tmp).float()
            result_y[i][class_id]  = 1.0

        if agumentation:
            result_x = self._augmentation(result_x)

        return result_x, result_y

    def _augmentation(self, x, gaussian_noise_level = 0.001, offset_noise_level = 1.0):
        noise        = gaussian_noise_level*torch.randn(x.shape)
        
        offset_noise = 2.0*torch.rand((x.shape[0], x.shape[1])).unsqueeze(2).repeat(1, 1, x.shape[2]) - 1.0
 
        x            = x + noise + offset_noise_level*offset_noise
        return x


if __name__ == "__main__":
    path = "/Users/michal/dataset/cells_dataset/sim26/"

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


    dataset = CellsDataset(training_files, training_labels, testing_files, testing_labels, classes_count = 2)

    x, y = dataset.get_training_batch()

    print(x.shape, y.shape)