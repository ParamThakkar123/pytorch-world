import torch
import torch.nn as nn
from world_models.training.train_dw import TrainTransformers
from world_models.configs.driving_world_config import DrivingWorldConfig


class DrivingWorld(nn.Module):
    """
    High-level wrapper for the Driving World model, providing a simple train() API.

    Usage example:
        from world_models.models.driving_world import DrivingWorld
        from world_models.configs.driving_world_config import DrivingWorldConfig

        config = DrivingWorldConfig()
        config.update(codebook_size=512, n_epochs=10)  # Modify as needed

        model = DrivingWorld(config)
        # Assume data_loader yields batches of (token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt)
        # and vqvae_codebook is a callable
        model.train(data_loader=your_data_loader, vqvae_codebook=your_codebook)
    """

    def __init__(self, config: DrivingWorldConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config.no_gpu else "cpu"
        )
        self.model = TrainTransformers(self.config).to(self.device)
        # Configure optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

    def train(self, data_loader, vqvae_codebook):
        """
        Train the model for config.n_epochs using the provided data_loader.

        Args:
            data_loader: Iterable yielding batches of (token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt).
                         Each should be tensors on the correct device.
            vqvae_codebook: Callable to map token indices to embeddings.
        """
        self.model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            for batch in data_loader:
                token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt = batch
                # Move to device if not already
                token = token.to(self.device)
                feature = feature.to(self.device)
                pose = pose.to(self.device)
                yaw = yaw.to(self.device)
                token_gt = token_gt.to(self.device)
                feature_gt = feature_gt.to(self.device)
                pose_gt = pose_gt.to(self.device)
                yaw_gt = yaw_gt.to(self.device)

                loss = self.model(
                    token,
                    feature,
                    pose,
                    yaw,
                    token_gt,
                    feature_gt,
                    pose_gt,
                    yaw_gt,
                    vqvae_codebook,
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.config.n_epochs}: Avg Loss {avg_loss:.4f}")

            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save(self.config.results_dir, epoch + 1)

    def evaluate(self, data_loader, vqvae_codebook, num_batches=10):
        """
        Evaluate the model on a few batches from data_loader.

        Args:
            data_loader: Iterable yielding batches.
            vqvae_codebook: Callable to map token indices to embeddings.
            num_batches (int): Number of batches to evaluate.

        Returns:
            dict: Average losses.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt = batch
                token = token.to(self.device)
                feature = feature.to(self.device)
                pose = pose.to(self.device)
                yaw = yaw.to(self.device)
                token_gt = token_gt.to(self.device)
                feature_gt = feature_gt.to(self.device)
                pose_gt = pose_gt.to(self.device)
                yaw_gt = yaw_gt.to(self.device)

                loss = self.model(
                    token,
                    feature,
                    pose,
                    yaw,
                    token_gt,
                    feature_gt,
                    pose_gt,
                    yaw_gt,
                    vqvae_codebook,
                    eval=True,
                )
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"Evaluation Avg Loss: {avg_loss:.4f}")
        return {"eval_loss": avg_loss}

    def generate(
        self,
        initial_token,
        initial_feature,
        initial_pose,
        initial_yaw,
        vqvae_codebook,
        steps=10,
    ):
        """
        Generate future predictions starting from initial conditions.

        Args:
            initial_token: Initial token tensor [B, T, H*W].
            initial_feature: Initial feature tensor [B, T, H*W, D].
            initial_pose: Initial pose tensor [B, T, 2].
            initial_yaw: Initial yaw tensor [B, T, 1].
            vqvae_codebook: Callable to map token indices to embeddings.
            steps (int): Number of steps to generate.

        Returns:
            dict: Generated tokens, features, poses, yaws.
        """
        self.model.eval()
        with torch.no_grad():
            # This assumes the model has a generate method; adjust based on TrainTransformers
            # Placeholder: call model's generate if available, else use forward in eval mode
            generated = self.model.generate_gt_pose_gt_yaw(
                initial_token,
                initial_feature,
                initial_pose,
                initial_yaw,
                vqvae_codebook,
                steps,
            )
            return generated

    def save(self, path, epoch):
        """Save the model state."""
        import os

        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/ckpt_{epoch}.pth")
        print(f"Model saved to {path}/ckpt_{epoch}.pth")

    def load(self, path, epoch):
        """Load the model state."""
        self.model.load_state_dict(
            torch.load(f"{path}/ckpt_{epoch}.pth", map_location=self.device)
        )
        print(f"Model loaded from {path}/ckpt_{epoch}.pth")

    def restore(self):
        """Restore from config.checkpoint_path if set."""
        if self.config.restore and self.config.checkpoint_path:
            self.load(self.config.checkpoint_path, epoch=None)
