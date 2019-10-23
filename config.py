############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 200
batch_size = 128
optim_type = 'SGD'

mean = {
    'cifar100': (0.5071, 0.4867, 0.4408)
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761)
}


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
