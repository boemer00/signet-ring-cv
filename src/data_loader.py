import os
import pandas as pd
from sklearn.model_selection import train_test_split

# remember to train, test split if you are training a new model
# I have included sklearn's method

def load_image_paths(base_dir, sub_dirs, file_ext='.jpeg'):
    """
    loads image paths from specif subdir and creates a df with image paths and labels

    :param base_dir: str, base directory containing the subdirectories
    :param sub_dirs: list of tuples, each tuple contains the subdirectory name and its associated label
    :param file_ext: str, file extension of the images
    :return: df, containing 2 columns ['image_path', 'label']
    """
    all_images = []
    for sub_dir, label in sub_dirs:
        dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(dir_path):
            print(f'Warning: Directory {dir_path} does not exist. Skipping.')
            continue
        images = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]
        df = pd.DataFrame(images, columns=['image_path'])
        df['label'] = label
        all_images.append(df)

    return pd.concat(all_images, ignore_index=True) if all_images else pd.DataFrame(columns=['image_path', 'label'])

def main():
    # base directory for images, relative to the project root
    base_img_dir = 'data'

    # subdirectories and their corresponding labels
    image_dirs = [('sig-train-neg', 0), ('sig-train-pos', 1)]

    # load images
    df = load_image_paths(base_img_dir, image_dirs)

    return df

if __name__ == "__main__":
    image_df = main()
    print(image_df.head())
