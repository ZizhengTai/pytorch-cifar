import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


class Shift3x3(nn.Module):
    def __init__(self, out_channels):
        super(Shift3x3, self).__init__()

        kernel = torch.zeros((out_channels, 1, 3, 3))
        channel_idx = 0
        for i in range(3):
            for j in range(3):
                num_channels = out_channels // 9
                if i == 1 and j == 1:
                    num_channels += out_channels % 9
                kernel[channel_idx:channel_idx+num_channels, 0, i, j] = 1
                channel_idx += num_channels

        self.out_channels = out_channels
        self._kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self._kernel, stride=1, padding=1,
                        groups=self.out_channels)


def make_shift_conv(conv):
    '''
    Given a conv3x3 layer, returns a (shift3x3, conv1x1) tuple.
    '''
    shift = Shift3x3(conv.out_channels)
    conv = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
    return shift, conv


def make_forward_hook(shift_conv):
    '''
    Given a (shift3x3, conv1x1) tuple, returns a forward hook
    that can be registered on the original conv3x3 layer to train
    the conv1x1 layer.
    '''
    critierion = nn.MSELoss()
    optimizer = optim.SGD(shift_conv[1].parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)

    def hook(module, input, target):
        optimizer.zero_grad()

        print(input[0].size())
        print(module.in_channels, module.out_channels)
        output = shift_conv[1](shift_conv[0](input[0]))
        loss = critierion(output, target)

        loss.backward()
        optimizer.step()

    return hook


def train(net, train_loader):
    '''
    Feed data through the network for one epoch.
    '''
    for input, _ in train_loader:
        input = Variable(input.cuda(), volatile=True)
        net(input)


def main():
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']

    # Use CUDA
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    # Freeze all parameters
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    # Create a (shift3x3, conv1x1) tuple for each conv3x3 layer,
    # and register forward hook to train the conv1x1
    shift_convs = []
    for mod in net.modules():
        if isinstance(mod, nn.Conv2d) and mod.kernel_size == (3, 3):
            shift_conv = make_shift_conv(mod)
            shift_convs.append(shift_conv)

            hook = make_forward_hook(shift_conv)
            mod.register_forward_hook(hook)

    # Prepare data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True,
                                              transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                               shuffle=True, num_workers=2)

    # Train
    for epoch in range(10):
        print('Epoch {0}'.format(epoch))
        train(net, train_loader)


if __name__ == '__main__':
    main()
