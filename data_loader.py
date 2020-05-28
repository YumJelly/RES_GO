from torch.utils.data import Dataset, DataLoader
import h5py


class H5Dataset(Dataset):
    def __init__(self, path):
        self.file_path = path
        self.xs = None
        self.ys = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["xs"])
        if self.xs is None:
            self.xs = h5py.File(self.file_path, 'r')["xs"]
        if self.ys is None:
            self.ys = h5py.File(self.file_path, 'r')["ys"]       
 
    def __getitem__(self, index):
        return (self.xs[index]/1.0, self.ys[index]/1.0) # devide 1.0 to change this to float 

    def __len__(self):
        return self.dataset_len



def load_data(batchSize):
    # load data for train & test
    train_set = H5Dataset('data/train.hdf5')
    test_set = H5Dataset('data/test.hdf5')
    training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize, pin_memory=True, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=batchSize, pin_memory=True, shuffle=True)
    return training_data_loader, testing_data_loader
