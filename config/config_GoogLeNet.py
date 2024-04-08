
# common conv
# [in_channels, out_channels, kernel_size, stride, padding]
conv1 = [3, 64, 7, 2, 3]
conv2 = [64, 64, 1, 1, 0]
conv3 = [64, 192, 3, 1, 1]

# inception module
# [in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj]
inception3a = [192, 64, 96, 128, 16, 32, 32]
inception3b = [256, 128, 128, 192, 32, 96, 64]

inception4a = [480, 192, 96, 208, 16, 48, 64]
inception4b = [512, 160, 112, 224, 24, 64, 64]
inception4c = [512, 128, 128, 256, 24, 64, 64]
inception4d = [512, 112, 144, 288, 32, 64, 64]
inception4e = [528, 256, 160, 320, 32, 128, 128]

inception5a = [832, 256, 160, 320, 32, 128, 128]
inception5b = [832, 384, 192, 384, 48, 128, 128]
