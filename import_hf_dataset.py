from datasets import load_dataset


DATASET_NAME = "DarthReca/south-africa-crop-type-clouds"


def load_crop_cloud_dataset():
    return load_dataset(DATASET_NAME)


if __name__ == "__main__":
    ds = load_crop_cloud_dataset()
    print(ds)
