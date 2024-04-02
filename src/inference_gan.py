import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.parse import opt
from utils.gan_model import Generator  
from utils.dataset_utils import OTS_test_loader  

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_model(model_path, device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference(val_loader, model, device):
    total_psnr = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            psnr = calculate_psnr(outputs, targets)
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    print(f'Average PSNR: {avg_psnr:.2f} dB')

def visualize_sample(data_loader, model, device, num_images=3):
    data, targets = next(iter(data_loader))
    data, targets = data.to(device), targets.to(device)
    with torch.no_grad():
        outputs = model(data)
    imgs = torch.cat([data[:num_images], outputs[:num_images], targets[:num_images]], dim=0)
    grid = make_grid(imgs, nrow=num_images)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title("Input - Generated - Target")
    plt.show()

if __name__ == '__main__':
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    model_path = 'best_generator_model.pth' 
    model = load_model(model_path, device)
    test_loader = OTS_test_loader  
    inference(test_loader, model, device)
    visualize_sample(test_loader, model, device)
