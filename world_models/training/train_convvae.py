import torch
from torchvision.utils import save_image
from world_models.datasets.wm_dataset import ObservationDataset
from world_models.vision.VAE import ConvVAE
from world_models.configs.wm_config import WMVAEConfig
from world_models.losses.convae_loss import conv_vae_loss_fn


def train_epoch(
    epoch: int, model, optimizer, train_loader, device, train_dataset, loss_fn
):
    model.train()
    train_loss = 0.0
    if hasattr(train_dataset, "load_next_buffer"):
        train_dataset.load_next_buffer()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        # transpose if dataset yields (B,H,W,C) -> (B,C,H,W)
        data = torch.transpose(data, 1, 3)
        optimizer.zero_grad()
        reconst, mu, logvar = model(data)
        loss = loss_fn(reconst, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                f"train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tloss: {loss.item() / len(data):.6f}"
            )

    print(
        "---> Epoch: {} Average Loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def train_convae(config: WMVAEConfig) -> None:
    device = torch.device(config.device if hasattr(config, "device") else "cpu")
    model = ConvVAE(img_channels=3, latent_size=config.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = ObservationDataset(root=config.data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )

    for ep in range(1, getattr(config, "num_epochs", 10) + 1):
        train_epoch(
            ep, model, optimizer, train_loader, device, train_dataset, conv_vae_loss_fn
        )
        if ep % 5 == 0:
            with torch.no_grad():
                sample = torch.randn(16, config.latent_size).to(device)
                sample = model.decoder(sample).cpu()
                save_image(sample.view(16, 3, 64, 64), f"results/sample_epoch_{ep}.png")

    torch.save(model.state_dict(), "convae_final.pth")
