from world_models.models.jepa_agent import JEPAAgent

if __name__ == "__main__":
    agent = JEPAAgent(
        dataset="cifar10",
        root_path=r"E:\pytorch-world\cifar",
        download=True,
        folder="results/cifar_jepa",
        write_tag="cifar_jepa",
        batch_size=16,
        pin_mem=False,
        crop_size=32,
        patch_size=4,
        enc_mask_scale=(0.05, 0.15),
        pred_mask_scale=(0.05, 0.15),
        min_keep=1,
        allow_overlap=True,
        num_workers=0,
        epochs=25,
    )
    agent.train()
