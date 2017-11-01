import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange


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
    shift = Shift3x3(conv.in_channels)
    conv = nn.Conv2d(conv.out_channels, conv.in_channels, 1, bias=False)
    return shift, conv


def make_forward_hook(shift, conv, rel_losses, index):
    '''
    Given a shift3x3 and a conv1x1, returns a forward hook that can be
    registered on the original conv3x3 layer to train the conv1x1 layer.
    '''
    critierion = nn.MSELoss()
    optimizer = SGD(conv.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    def hook(module, input, target):
        optimizer.zero_grad()

        output = conv(shift(input[0]))
        loss = critierion(output, target)

        rel_loss = (np.prod(target.size()) * loss.data[0] /
                    (target.norm() ** 2).data[0])
        rel_losses[index] = rel_loss

        loss.backward()
        optimizer.step()

    return hook, scheduler


def train(net, train_loader, conv1x1_losses, schedulers):
    '''
    Feed data through the network for one epoch.
    '''
    # Adjust learning rate
    for scheduler in schedulers:
        scheduler.step()

    t = tqdm(train_loader)
    for input, _ in t:
        input = Variable(input.cuda())
        net(input)

        losses = np.array(conv1x1_losses)
        t.set_description('avg: {0:.3e} | min: {1:.3e} | max: {2:.3e}'.format(
            losses.mean(), losses.min(), losses.max()))


def save_losses(losses, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, losses)))
        f.write('\n')


def save_modules(modules, filename):
    torch.save(modules, filename)


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
    modules = []
    conv1x1_losses = []
    schedulers = []
    for mod in net.modules():
        if (isinstance(mod, nn.Conv2d) and
            mod.kernel_size == (3, 3) and
            mod.in_channels > 9 and
            mod.stride == (1, 1)):

            # Create shift3x3 and conv1x1 on GPU
            shift3x3, conv1x1 = make_shift_conv(mod)
            shift3x3, conv1x1 = shift3x3.cuda(), conv1x1.cuda()
            modules.append(nn.Sequential(shift3x3, conv1x1))
            conv1x1_losses.append(0)

            hook, scheduler = make_forward_hook(
                    shift3x3, conv1x1, conv1x1_losses, len(conv1x1_losses) - 1)
            mod.register_forward_hook(hook)
            schedulers.append(scheduler)

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
    n_epochs = 100
    for epoch in range(n_epochs):
        print('Epoch {0}/{1}'.format(epoch, n_epochs))

        train(net, train_loader, conv1x1_losses, schedulers)

        print('Saving...')
        save_modules(modules, 'saves/modules.pth')
        save_losses(conv1x1_losses, 'saves/losses.csv')
        print()


if __name__ == '__main__':
    main()
