import numpy


class DatsLoad:
    def __init__(self, files_list, cols):
        data        = []
        shortest    = 10**9

        for f in files_list:
            print("loading ", f)
            data_   = numpy.genfromtxt(f, skip_header=1,  unpack = True)
            data_   = numpy.array(data_, dtype=float)

            data_   = data_[cols]

            if (data_.shape[1] < shortest):
                shortest = data_.shape[1]

            print("shape = ", data_.shape)

            data.append(data_)

        self.data = numpy.zeros((len(data), len(cols), shortest))

        for i in range(len(data)):
            for j in range(len(cols)):
                self.data[i][j] = data[i][j][0:shortest]

        self.data      = numpy.array(self.data, dtype=float)
        self.data      = numpy.rollaxis(self.data, 2, 1)

        print("shortest_length = ", shortest)
        print("data_shape      = ", self.data.shape)
        print("\n\n")

        #data normalisation
        cells_count = self.data.shape[0]
        time_steps  = self.data.shape[1]
        axis_count  = self.data.shape[2]

        self.data      = numpy.reshape(self.data, (cells_count*time_steps, axis_count))

        mean = self.data.mean(axis=0)
        std  = self.data.std(axis=0)

        self.data = (self.data - mean)/std
        
        '''
        print("shape    = ", self.data.shape)
        print("min      = ", self.data.min (axis=0))
        print("max      = ", self.data.max(axis=0))
        print("mean     = ", self.data.mean(axis=0))
        print("std      = ", self.data.std(axis=0))
        '''

        self.data      = numpy.reshape(self.data, (cells_count, time_steps, axis_count))

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


    dats = DatsLoad(files_list, cols = [1, 2, 3]) 