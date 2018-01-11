import torchvision.transforms as transforms

def default_image_tf(scale_size, crop_size,
        mean=[0.485, 0.456, 0.406], # resnet imagnet
        std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Scale(scale_size),
        transforms.RandomCrop(crop_size),
        #transforms.CenterCrop(size),
        transforms.ToTensor(), # divide by 255 automatically
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform
