import os
import random

import matplotlib.pyplot as plt


def inspect_image(dataset_name, dataloaders=None, class_to_idx=None, class_to_name=None, image_idx=None, plot=False):
    """
    Inspect an image in a pytorch dataloader dataset.

    Selects random image if no index is passed, otherwise selects image at given image_index.

    Plots image if plot=True

    Returns metadata as a dict
    """
    if dataloaders is None or dataset_name is None or class_to_idx is None or class_to_name is None:
        raise ValueError('Missing argument for dataloaders, dataset_name, class_to_name, or class_to_idx')

    dataset = dataloaders[dataset_name].dataset

    if image_idx is not None:
        image_data, image_label = dataset[image_idx]
        image_path, _ = dataset.samples[image_idx]

    else:
        image_idx = random.randint(0, len(dataset) - 1)
        image_data, image_label = dataset[image_idx]
        image_path, _ = dataset.samples[image_idx]

    class_ = list(class_to_idx.keys())[list(class_to_idx.values()).index(image_label)]
    class_name = class_to_name.get(class_, 'Unknown')

    metadata = {
        'class': class_,
        'class_index': class_to_idx[class_],
        'class_name': class_name,
        'image_path': image_path,
        'image_index': image_idx
    }

    if plot:
        image_data_clamped = image_data.clamp(0, 1)  # Ensure image data is in the range [0,1]
        plt.imshow(image_data_clamped.permute(1, 2, 0))

        title_text = f"Class (Idx): {metadata['class']} ({metadata['class_index']}) | Name: {metadata['class_name']} | Index: {metadata['image_index']}"
        subtitle_text = f'{image_path}'

        plt.title(f'{title_text}\n{subtitle_text}', fontsize=10)
        plt.show()

    return metadata
