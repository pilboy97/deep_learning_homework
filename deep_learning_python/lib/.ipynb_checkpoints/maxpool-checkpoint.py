import cupy as np
class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size
        
        for i in range(new_h):
            for j in range(new_w):
                img_region = image[(i*self.pool_size):(i*self.pool_size+self.pool_size),
                                   (j*self.pool_size):(j*self.pool_size+self.pool_size)]
                yield img_region, i, j
    
    def forward(self, input):
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        
        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(img_region, axis=(0, 1))
        
        return output
    
    def backward(self, d_L_d_out):
        return None
