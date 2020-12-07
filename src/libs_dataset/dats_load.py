import numpy

class DatsLoad:
    def __init__(self, files_list, cols):
        self.data      = []

        shortest = 10**9

        for f in files_list:
            print("loading ", f)
            data_   = numpy.genfromtxt(f, skip_header=1,  unpack = True)
            data_   = numpy.array(data_, dtype=float)

            data_   = data_[cols]

            if (data_.shape[1] < shortest):
                shortest = data_.shape[1]

            print("shape = ", data_.shape)

            self.data.append(data_)

        print("shortest_length = ", shortest, "\n\n")

        for i in range(len(self.data)):
            self.data[i] = self.data[i][0:][0:shortest]


        self.data      = numpy.array(self.data, dtype=float)
        self.data      = numpy.rollaxis(self.data, 2, 1)
      

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