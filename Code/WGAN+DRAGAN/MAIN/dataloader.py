from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#loads data
def dataloader(dataset, input_size, batch_size, split='train'):
    #for color images
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        #if dataset is mnist, then we need to make our data black/white
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar100':
        data_loader = DataLoader(
            datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader
