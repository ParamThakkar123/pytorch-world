import torch
import numpy as np
import cv2
from torchwm.configs.iris_config import IRISConfig
from torchwm.models.iris_agent import IRISAgent
from torchwm.envs.ale_atari_env import make_atari_env


def preprocess_frame(frame, size=64):
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    return frame.transpose(2, 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = IRISConfig()
    env = make_atari_env(
        "ALE/Breakout-v5", obs_type="rgb", frameskip=4, render_mode="human"
    )
    action_size = env.action_space.n

    agent = IRISAgent(config=config, action_size=action_size, device=device)
    agent.load("checkpoints/iris/best_Breakout-v5.pt")
    agent.eval()
    print("Model loaded!")

    num_episodes = 100
    total_reward = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = preprocess_frame(obs)
        episode_reward = 0
        steps = 0

        print(f"\n--- Episode {ep + 1} ---")

        while True:
            env.render()

            frame_tensor = (
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action = agent.act(frame_tensor, epsilon=0.0, temperature=0.5).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

            if not done:
                obs = preprocess_frame(obs)
            else:
                break

        total_reward += episode_reward
        print(f"Episode {ep + 1}: Reward = {episode_reward}, Steps = {steps}")

    env.close()
    print(
        f"\nAverage reward over {num_episodes} episodes: {total_reward / num_episodes:.2f}"
    )


if __name__ == "__main__":
    main()
