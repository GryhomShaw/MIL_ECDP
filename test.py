import numpy as np
import torch
import torch.utils.data as data

class testdataset(data.Dataset):
    def __init__(self,len,s):

        self.len = len
        self.s = s
        grid = []
        temp = []
        for i in range(len):
            for j in range(s):
                temp.append(i*self.s + j)
            grid.append(temp)
            temp = []
        #print(grid)
        self.grid = grid
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        grid = self.grid[index]
        return grid



if __name__ == '__main__':
    dataset = testdataset(10,2)
    loader = torch.utils.data.DataLoader(dataset,batch_size=5,shuffle=False)
    for i, input in enumerate(loader):
        print(len(input))
        print(input[0],input[1])