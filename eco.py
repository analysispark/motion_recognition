

class BasicConv(nn.module):
    '''ECO의 2D Net 모듈의 첫 번째 모듈'''

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(
            63, 63, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(
                192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

    def forward(self, x):
        out = self.conv1_7x7_s2(x)
        out = self.conv1_7x7_s2_bn(out)
        out = self.conv1_relu_7x7(out)
        out = self.pool1_3x3_s2(out)
        out = self.conv2_3x3_reduce(out)
        out = self.conv2_3x3_reduce_bn(out)
        out = self.conv2_relu_3x3_reduce(out)
        out = self.conv2_3x3(out)
        out = self.conv2_3x3_bn(out)
        out = self.conv2_relu_3x3(out)
        out = self.pool2_3x3_s2(out)
        return out

class InceptionA(nn.Module):
    '''InceptionA'''

    def __init__(self) -> None:
        super(InceptionA, self).__init__()

        self.inception_3a_1x1 = nn.Conv2d(
                192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace=True)

        self.inception_3a_3x3_reduece = nn.Conv2d(
                192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduece_bn = nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3_reduece = nn.ReLU(inplace=True)
        self.inception_3a_3x3 = nn.Conv2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace=True)
        ## 496p 부터 작성 필요
