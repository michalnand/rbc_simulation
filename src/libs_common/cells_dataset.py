import numpy
import torch
from dats_load import *

class CellsDataset:
    def __init__(self, files_list, cols = [1, 2, 3], prediction_steps = 50, window_size = 32):
        dats = DatsLoad(files_list, cols)

        self.data = self._normalise(dats.data)
        self.difference_steps = 100
        self.averaging_steps  = 16

        self.prediction_steps   = prediction_steps
        self.window_size        = window_size

        self.time_steps  = self.data.shape[0]
        self.axis_count  = self.data.shape[1]
        self.cells_count = self.data.shape[2]
        
        self.input_shape  = (self.axis_count + 1, self.cells_count, self.window_size)
        self.output_shape = (self.axis_count, )

        self.time_steps_min = self.window_size + self.difference_steps + self.averaging_steps
        self.time_steps_max = self.time_steps - self.prediction_steps - 1
        
        print("data  shape      = ", self.data.shape)
        print("input shape      = ", self.input_shape)
        print("output shape     = ", self.output_shape)
        print("total time steps = ", self.time_steps)
        print("axis count       = ", self.axis_count)
        print("cells count      = ", self.cells_count)
        print("\n")
        print("mean = ", self.data.mean())
        print("std  = ", self.data.std())
        print("min  = ", self.data.min())
        print("max  = ", self.data.max())
        print("\n")
       
    


    def get(self, cell_idx, time_idx):
        input    = self._get_input(cell_idx, time_idx)
        output   = self._get_output(cell_idx, time_idx + self.prediction_steps)
        return input, output
    
    def get_random(self):
        cell_idx = numpy.random.randint(self.cells_count)
        time_idx = numpy.random.randint(self.time_steps_min, self.time_steps_max)
        return self.get(cell_idx, time_idx)


    def get_batch(self, batch_size = 256):
        input_t   = torch.zeros((batch_size, ) + self.input_shape)
        output_t  = torch.zeros((batch_size, ) + self.output_shape)


        for b in range(batch_size):
            input, output = self.get_random()
            input[b]  = torch.from_numpy(input)
            output[b] = torch.from_numpy(output)


        return input, output



    def _normalise(self, data):
        mean = data.mean()
        std  = data.std()

        return (data - mean)/std

    def _get_input(self, cell_idx, time_idx):
        cell_input = self.data[time_idx - self.window_size:time_idx]
        cell_input = numpy.rollaxis(cell_input, 0, 2)
        cell_input = numpy.rollaxis(cell_input, 2, 1)

        cell_mask = numpy.zeros((1, self.input_shape[1], self.input_shape[2]))
        cell_mask[0][cell_idx] = numpy.ones(self.input_shape[2])

        result_input  = numpy.concatenate((cell_input, cell_mask), axis = 0)

        print("cell_idx = ", cell_idx)
        print("time_idx = ", time_idx)
        print("tensor_shape = ", result_input.shape)
        print(numpy.round(result_input, 4))
        
        return result_input

    def _get_output(self, cell_idx, time_idx):

        difference      = self.difference_steps
        averaging_steps = self.averaging_steps
        
        position_prev   = self.data[time_idx-averaging_steps-difference:time_idx-difference][:]
        position        = self.data[time_idx-averaging_steps:time_idx][:]

        position_prev   = numpy.rollaxis(position_prev, 2, 0)[cell_idx]
        position        = numpy.rollaxis(position, 2, 0)[cell_idx]

        dif = position - position_prev

        result = dif.mean(axis=0)

        print("target_output = ", result)
            
        return result



if __name__ == "__main__":
    path = "/Users/michal/dataset/cells_dataset/sim26/"

    files_list = []
    files_list.append(path + "rbc0_data_sim26.dat")
    files_list.append(path + "rbc1_data_sim26.dat")
    files_list.append(path + "rbc2_data_sim26.dat")
    files_list.append(path + "rbc3_data_sim26.dat")
    files_list.append(path + "rbc4_data_sim26.dat")
    files_list.append(path + "rbc5_data_sim26.dat")
    files_list.append(path + "rbc6_data_sim26.dat")


    dats = CellsDataset(files_list)

    input, output = dats.get_random()