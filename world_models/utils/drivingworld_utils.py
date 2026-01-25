import torch
import math
import torch.nn.functional as F
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
import cv2
import numpy as np
import logging
import os
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import imageio
import json

logger = logging.getLogger("base")


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 100,
):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param embedding_dim: the dimension of the output.
    :param flip_sin_to_cos: if True, use cosine for even indices instead of sine.
    :param scale: a scaling factor for the timesteps.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x embedding_dim] Tensor of positional embeddings.
    """

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2

    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


def get_fourier_embeds_from_coordinates(
    embed_dim,
    coordinates,
    max_period: int = 100,
):
    """
    Args:
        embed_dim: int
        coordinates: a tensor [B x N x C] representing the coordinates of N points in C dimensions
    Returns:
        [B x N x C x embed_dim] tensor of positional embeddings
    """
    half_embed_dim = embed_dim // 2
    emb = max_period ** (
        torch.arange(half_embed_dim, dtype=torch.float32, device=coordinates.device)
        / half_embed_dim
    )
    emb = emb[None, None, None, :] * coordinates[:, :, :, None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


def top_k_sampling(logits, top_k=30, temperature_k=1.0):
    """
    Sampling from top-k.
    In LLM, lowering the temperature can reduce the semantic inconsistency problem in long-context generation.
    Args:
        logits: [B, 1, N], N is the codebook size.
    """
    assert len(logits.shape) == 3
    logits_topk = logits / temperature_k
    v, _ = torch.topk(logits_topk, min(top_k, logits_topk.size(-1)))
    logits_topk[logits_topk < v[:, :, [-1]]] = -float("Inf")
    probs = F.softmax(logits_topk, dim=-1)
    idx_next = torch.multinomial(probs[:, 0, :], num_samples=1)
    return idx_next


def top_p_sampling(logits, top_p=0.9, temperature_p=1.0, filter_value=-float("Inf")):
    """
    Keep the top tokens with cumulative probability >= top_p (nucleus filtering), see https://arxiv.org/abs/1904.09751.
    In LLM, lowering the temperature can reduce the semantic inconsistency problem in long-context generation.
    Args:
        logits: [B, 1, N], N is the codebook size.
    """
    B, _, N = logits.shape
    logits_top_p = logits[:, 0, :] / temperature_p
    sorted_logits, sorted_indices = torch.sort(logits_top_p, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    for i in range(B):
        indices_to_remove = sorted_indices[i : i + 1, ...][
            sorted_indices_to_remove[i : i + 1, ...]
        ]
        logits_top_p[i : i + 1, indices_to_remove] = filter_value

    probabilities = F.softmax(logits_top_p, dim=-1)
    idx_next = torch.multinomial(probabilities, 1)
    return idx_next


def pk_sampling(
    logits,
    top_k=30,
    temperature_k=1.0,
    top_p=0.9,
    temperature_p=1.0,
    filter_value=-float("Inf"),
):
    assert len(logits.shape) == 3
    logits_topk = logits / temperature_k
    v, _ = torch.topk(logits, min(top_k, logits_topk.size(-1)))
    logits_topk[logits_topk < v[:, :, [-1]]] = -float("Inf")
    # apply softmax to convert logits to (normalized) probabilities
    logits = F.softmax(logits_topk, dim=-1)

    B, _, N = logits.shape
    logits_top_p = logits[:, 0, :] / temperature_p
    sorted_logits, sorted_indices = torch.sort(logits_top_p, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    for i in range(B):
        indices_to_remove = sorted_indices[i : i + 1, ...][
            sorted_indices_to_remove[i : i + 1, ...]
        ]
        logits_top_p[i : i + 1, indices_to_remove] = filter_value

    probabilities = F.softmax(logits_top_p, dim=-1)
    idx_next = torch.multinomial(probabilities, 1)
    return idx_next


## Rope functions
def init_t_xy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t.remainder(end_x)).float()
    t_y = t.div(end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim, end_x, end_y, theta=100.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x, freqs_y = torch.outer(t_x, freqs), torch.outer(t_y, freqs)
    freqs_cis_x, freqs_cis_y = torch.polar(
        torch.ones_like(freqs_x), freqs_x
    ), torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1).cuda()


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim > 1, "Input tensor x must have at least 2 dimensions."
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 3) + [x.shape[-3], x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[1], x.shape[-1]):
        shape = [1] * ndim
        shape[1] = x.shape[1]
        shape[-1] = x.shape[-1]
    else:
        raise ValueError("Shape of freqs_cis does not match x in any expected pattern.")
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_complex, xk_complex = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], -1, 2)
    ), torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_rotated, xk_rotated = torch.view_as_real(
        xq_complex * freqs_cis_broadcast
    ).flatten(-2), torch.view_as_real(xk_complex * freqs_cis_broadcast).flatten(-2)
    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)


## Merge Dataset functions
class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self,
        src_dataset_ls,
        batch_size,
        rank,
        seed,
        num_replicas,
        drop_last=True,
        shuffle=True,
        prob=None,
        generator=None,
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.prob_generator = torch.Generator().manual_seed(seed + rank * seed)

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length
        # for dist
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        # end dist

        # BatchSamplers for each source dataset
        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(
                        ds, replacement=False, generator=self.generator
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset

        # start dist
        self.raw_batches_split = []
        for i in range(len(self.raw_batches)):
            batch = self.raw_batches[i]
            self.raw_batches_split.append(
                list(batch[self.rank : self.dataset_length[i] : self.num_replicas])
            )

        self.n_batches = [len(b) for b in self.raw_batches_split]
        self.n_total_batch = sum(self.n_batches)
        # end dist

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)

    def __iter__(self):
        """_summary_

        Yields:
            list(int): a batch of indics, corresponding to ConcatDataset of src_dataset_ls
        """
        for i in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob,
                1,
                replacement=True,
                generator=self.prob_generator,
            ).item()

            # if batch list is empty, generate new list
            if 0 == len(self.raw_batches_split[idx_ds]):
                # self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
                self.raw_batches_split[idx_ds] = list(
                    self.raw_batches[idx_ds][
                        self.rank : self.dataset_length[idx_ds] : self.num_replicas
                    ]
                )
            # get a batch from list
            batch_raw = self.raw_batches_split[idx_ds].pop()
            # shift by cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch


## Undistort image
def get_undistort_map(
    img_shape,
    cam_in=[
        [1905.89892578125, 0.0, 1918.4837646484375],
        [0.0, 1905.89892578125, 1075.7330322265625],
        [0.0, 0.0, 1.0],
    ],
    cam_dist_coeffs=[
        0.8017038106918335,
        0.10657747834920883,
        1.5339870742536732e-06,
        -7.50786193748354e-06,
        0.0010572359897196293,
        1.1659871339797974,
        0.30344289541244507,
        0.011946137063205242,
    ],
):
    """
    Rectify camera distortions.
    Camera intrinsic parameters and distortions examples are as follows.
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

    dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Depending on the distortion model used, this array might have fewer or more values.
    Args:
        cam_in: camera intrinsic parameters, [fx, fy, cx, cy]
        cam_dist_coeffs: camera distortion parameters, [k1, k2, p1, p2, k3]

    """
    # Get the dimensions of the image
    h, w = img_shape
    camera_matrix = np.array(cam_in)
    dist_coeffs = np.array(cam_dist_coeffs)

    # Get the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # # Undistort the image
    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Map coordinates from new camera matrix to distorted image using initUndistortRectifyMap which inversely remaps for distortion
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5
    )
    return [map1, map2, roi]


## Flow and time functions
def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(
    logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False
):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (
        normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1]
    )
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def rgb2ycbcr(img_np):
    h, w, _ = img_np.shape
    Y = 0.257 * img_np[:, :, 2] + 0.504 * img_np[:, :, 1] + 0.098 * img_np[:, :, 0] + 16

    return Y


## Running functions
def init_optimizer(model, lr=1e-4, weight_decay=1e-3):
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5):
    scheduler = MultiStepLR(optimizer, milestones=milstones, gamma=gamma)
    return scheduler


def save_model(self, path, epoch, rank=0):
    if rank == 0:
        torch.save(self.model.state_dict(), "{}/tvar_{}.pkl".format(path, str(epoch)))


def save_ckpt(
    args, path, model, optimizer=None, scheduler=None, curr_iter=0, curr_epoch=None
):
    """
    Save the model, optimizer, lr scheduler.
    """

    ckpt = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        # scheduler=scheduler.state_dict(),
    )

    ckpt_path = "{}/tvar_{}.pkl".format(path, str(curr_iter))

    torch.save(ckpt, ckpt_path)

    print(f"#### Save model: {ckpt_path}")


def resume_ckpt(local_rank, args, model, optimizer=None):

    resume_load_path = "{}/tvar_{}.pkl".format(
        args.save_model_path, str(args.resume_step)
    )
    print(local_rank, ": loading...... ", resume_load_path)
    ckpt_file = torch.load(resume_load_path, map_location="cpu")

    if "optimizer_state_dict" in ckpt_file:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt_file["optimizer_state_dict"])
        print(
            local_rank,
            f"Rank: {local_rank}, Successfully loaded optimizer from {resume_load_path}.",
        )
    if "model_state_dict" in ckpt_file:
        # print('loaded weight, model.pose_emb.weight sum:', torch.sum(ckpt_file['model_state_dict']['pose_emb.weight']))
        # model.load_state_dict(ckpt_file['model_state_dict'], strict=True)
        model = load_parameters(model, ckpt_file)
        print(
            local_rank,
            f"Rank: {local_rank}, Successfully loaded model from {resume_load_path}.",
        )
    else:
        # model.load_state_dict(ckpt_file, strict=False)
        model = load_parameters(model, ckpt_file)
        print(
            local_rank,
            f"Rank: {local_rank}, Successfully loaded model from {resume_load_path}.",
        )
    return model, optimizer


def load_parameters(model, load_ckpt_file):
    if "model_state_dict" in load_ckpt_file:
        ckpt = load_ckpt_file["model_state_dict"]
    else:
        ckpt = load_ckpt_file
    ckpt_state_dict = {}
    for key, val in ckpt.items():
        if key in model.state_dict() and val.shape == model.state_dict()[key].shape:
            ckpt_state_dict[key] = val
        elif key not in model.state_dict():
            print(f"!!!! {key} not exists in model.")
            # ckpt_state_dict[key] = val
            continue
        elif val.shape != model.state_dict()[key].shape:
            print(
                f"!!!! Shape of ckpt's {key} is {val.shape}, but model's shape is {model.state_dict()[key].shape}"
            )
            if key == "pos_emb":
                ckpt_state_dict[key] = model.state_dict()[key]
                B, H, W = val.shape
                B1, H1, W1 = ckpt_state_dict[key].shape
                B, H, W = min(B, B1), min(H, H1), min(W, W1)
                ckpt_state_dict[key][:B, :H, :W] = val[:B, :H, :W]
                print(f"!!!! load {B} {H} {W}")
            elif key == "img_projector.0.weight":
                ckpt_state_dict[key] = torch.zeros_like(model.state_dict()[key])
                H, W = val.shape
                H1, W1 = ckpt_state_dict[key].shape
                H, W = min(H, H1), min(W, W1)
                ckpt_state_dict[key][:H, :W] = val[:H, :W]
                print(f"!!!! load {H} {W}")
            else:
                print(f"!!!! no weight loaded for {key}")
                ckpt_state_dict[key] = model.state_dict()[key]
    newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(
        list(ckpt.keys())
    )
    for key in newparas_not_in_ckpt:
        print(
            f"!!!! {key} required by the model does not exist in ckpt. Shape is {model.state_dict()[key].shape}"
        )
        ckpt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(ckpt_state_dict, strict=True)
    return model


def log_dict_args(dict_args):
    """
    Logs a dictionary of arguments.

    Args:
        dict_args (dict): The dictionary of arguments to be logged.
    """
    # Convert the dictionary to a JSON string
    json_args = json.dumps(dict_args, indent=4)

    # Log the JSON string
    logging.info(f"Arguments: {json_args}")


## Testing functions
def add_border(img, border_size=10, value=[255, 255, 255]):
    bordered_img = cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )
    return bordered_img


def set_text(image, pose):
    # pose: string
    number = pose
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        number, font, font_scale, thickness
    )
    x = width - text_width - 10
    y = text_height + 10
    print(number)
    image = cv2.putText(image, number, (x, y), font, font_scale, color, thickness)
    return image


def create_video(args, imgs, video_save_path, border_size=10):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    fps = 2
    condition_frames = args.condition_frames

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape
    video_writer = cv2.VideoWriter(
        os.path.join(video_save_path, "video.mp4"),
        fourcc,
        fps,
        (w + 2 * border_size, h + 2 * border_size),
    )

    for j, image_file in enumerate(imgs):
        img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[
            :, :, ::-1
        ]
        if j < condition_frames:
            bordered_img = add_border(img, border_size=border_size)
        else:
            bordered_img = add_border(img, border_size=border_size, value=[0, 0, 255])
        cv2.imwrite(os.path.join(video_save_path, "%d.png" % (j)), bordered_img)
        video_writer.write(bordered_img)

    video_writer.release()


def create_mp4(args, imgs, video_save_path, border_size=10, fps=2):
    condition_frames = args.condition_frames
    print("save fps as ", fps)

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape

    # video_writer = cv2.VideoWriter(os.path.join(video_save_path, 'video.mp4'), fourcc, fps, (w+2*border_size, h+2*border_size))
    with imageio.get_writer(
        os.path.join(video_save_path, "video.mp4"), mode="I", fps=fps
    ) as writer:
        for j, image_file in enumerate(imgs):
            img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[
                :, :, ::-1
            ]
            if j < condition_frames:
                bordered_img = add_border(img, border_size=border_size)
            else:
                bordered_img = add_border(
                    img, border_size=border_size, value=[0, 0, 255]
                )
            # cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), bordered_img)
            writer.append_data(bordered_img[:, :, ::-1])


def create_gif(args, imgs, video_save_path, border_size=10, fps=2):
    images = []

    condition_frames = args.condition_frames

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    _, _, h, w = imgs[0].shape

    # video_writer = cv2.VideoWriter(os.path.join(video_save_path, 'video.mp4'), fourcc, fps, (w+2*border_size, h+2*border_size))
    for j, image_file in enumerate(imgs):
        img = (image_file[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")[
            :, :, ::-1
        ]
        if j < condition_frames:
            bordered_img = add_border(img, border_size=border_size)
        else:
            bordered_img = add_border(img, border_size=border_size, value=[0, 0, 255])
        cv2.imwrite(os.path.join(video_save_path, "%d.png" % (j)), bordered_img)
        images.append(bordered_img[:, :, ::-1])

    imageio.mimsave(os.path.join(video_save_path, "vis.gif"), images, fps=fps)
