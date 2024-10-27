import numpy as np

class Conv2D:
    def __init__(self, input_channels, output_channels, filter_size, stride=1, padding=1):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        self.W = np.random.randn(output_channels, input_channels, filter_size, filter_size) * np.sqrt(2.0 / input_channels)
        self.b = np.zeros(output_channels)

        self.X = None

    def forward(self, X):
        self.X = X
        
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        
        batch_size, input_channels, input_height, input_width = X_padded.shape
        kernel_height, kernel_width = self.filter_size, self.filter_size

        output_height = (input_height - kernel_height) // self.stride + 1
        output_width = (input_width - kernel_width) // self.stride + 1

        output = np.zeros((batch_size, self.output_channels, output_height, output_width))

        W_reshaped = self.W.reshape(self.output_channels, -1).T

        for i in range(output_height):
            for j in range(output_width):
                i_start = i * self.stride
                i_end = i_start + kernel_height
                j_start = j * self.stride
                j_end = j_start + kernel_width

                X_slice = X_padded[:, :, i_start:i_end, j_start:j_end].reshape(batch_size, -1)

                output[:, :, i, j] = np.tensordot(X_slice, W_reshaped, axes=1).reshape(batch_size, self.output_channels) + self.b

        return output

    
    def backward(self, d_out):
        batch_size, input_channels, input_height, input_width = self.X.shape
        filter_height, filter_width = self.filter_size, self.filter_size
        _, _, output_height, output_width = d_out.shape

        X_padded = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.W)
        db = np.sum(d_out, axis=(0, 2, 3))

        for i in range(output_height):
            for j in range(output_width):
                X_slice = X_padded[:, :, i*self.stride:i*self.stride+filter_height, j*self.stride:j*self.stride+filter_width]
                for k in range(self.output_channels):
                    dW[k] += np.tensordot(d_out[:, k, i, j], X_slice, axes=(0, 0))
                    dX_padded[:, :, i*self.stride:i*self.stride+filter_height, j*self.stride:j*self.stride+filter_width] += d_out[:, k, i, j].reshape(-1, 1, 1, 1) * self.W[k]

        dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else dX_padded

        self.dW = dW
        self.db = db

        return dX
