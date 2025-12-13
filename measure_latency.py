import torch
import time
from torchvision import models

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).to(device).eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    iters = 100
    elapsed = []
    for _ in range(iters):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)

    mean = sum(elapsed) / len(elapsed)
    std = torch.tensor(elapsed).std().item()
    output = f"Device: {device.type}\nMean latency: {mean:.2f} ms\nStd latency: {std:.2f} ms\n"
    print(output.strip())
    with open('latency_stats.txt', 'w', encoding='utf-8') as fh:
        fh.write(output)

if __name__ == '__main__':
    main()
