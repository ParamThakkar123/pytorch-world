from torch.utils.data import ConcatDataset
from world_models.datasets.nuplan.nuplan_demo_dataset import DemoTest
from world_models.datasets.nuplan.nuplan_dataset import NuPlanTest


def create_test_datasets(args):
    data_list = args.test_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name == "demo":
            dataset = DemoTest(
                args.datasets_paths["demo_root"],
                condition_frames=args.condition_frames,
            )
        elif data_name == "nuplan":
            dataset = NuPlanTest(
                args.datasets_paths["nuplan_root"],
                args.datasets_paths["nuplan_json_root"],
                condition_frames=args.condition_frames,
            )
        dataset_list.append(dataset)
    data_array = ConcatDataset(dataset_list)
    return data_array
