import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # x: [B,1,28,28] -> [B,784]
        return self.net(x.view(-1, 784))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FCNet().to(DEVICE)
checkpoint = torch.load("backdoored_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

print("Модель успешно загружена.")


transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

idx = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[idx]

# image: [1,28,28] -> [1,1,28,28]
image = image.unsqueeze(0).to(DEVICE)
label_tensor = torch.tensor([label], dtype=torch.long, device=DEVICE)

print(f"Случайный индекс: {idx}")
print(f"Истинная метка: {label}")


with torch.no_grad():
    logits = model(image)
    probs = F.softmax(logits, dim=1)
    pred_before = logits.argmax(dim=1).item()
    conf_before = probs[0, pred_before].item()

print(f"До атаки: pred={pred_before}, confidence={conf_before:.4f}")


def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)

    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.detach()
    signed_grad = grad.sign()

    x_adv = x_adv + epsilon * signed_grad
    x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv, grad, signed_grad


EPSILON = 0.09
adv_image, grad, signed_grad = fgsm_attack(model, image, label_tensor, EPSILON)



with torch.no_grad():
    adv_logits = model(adv_image)
    adv_probs = F.softmax(adv_logits, dim=1)
    pred_after = adv_logits.argmax(dim=1).item()
    conf_after = adv_probs[0, pred_after].item()

print(f"После атаки: pred={pred_after}, confidence={conf_after:.4f}")



perturbation = adv_image - image



def to_np(t):
    return t.detach().cpu().squeeze().numpy()

orig_np = to_np(image)
grad_np = to_np(grad)
sign_np = to_np(signed_grad)
pert_np = to_np(perturbation)
adv_np = to_np(adv_image)

plt.figure(figsize=(15, 3))

plt.subplot(1, 5, 1)
plt.imshow(orig_np, cmap="gray")
plt.title(f"Original\ntrue={label}, pred={pred_before}")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(grad_np, cmap="gray")
plt.title("Gradient dL/dx")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(sign_np, cmap="gray")
plt.title("sign(gradient)")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(pert_np, cmap="gray")
plt.title(f"Perturbation\nε={EPSILON}")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(adv_np, cmap="gray")
plt.title(f"Adversarial\npred={pred_after}")
plt.axis("off")

plt.tight_layout()
plt.show()