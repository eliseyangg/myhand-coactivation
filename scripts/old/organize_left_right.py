import os
import shutil
from coactivation.dataset_definition import left_right_hand_dict


if __name__ == "__main__":
    plot_dir = 'plots'

    base_dir = 'plots_left_right'
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Create or clear the specific save directory
    for new_dir in ['unknown','left','right']:
        if not os.path.exists(os.path.join(base_dir, new_dir)):
            os.makedirs(os.path.join(base_dir, new_dir))

    for root, subdirs, files in os.walk(plot_dir):
        print(plot_dir)
        if root != plot_dir:
            print(f"\nIn subdirectory: {root}")
            if files: 
                subject_id = root.split('/')[-1].split('_')[-1]
                if subject_id not in left_right_hand_dict:
                    folder = 'unknown'
                else:
                    if left_right_hand_dict[subject_id] == "left":
                        folder = 'left'
                    else: 
                        folder = 'right'
                dst = os.path.join(base_dir, folder) + '/' + root.split('/')[-1]

                print(dst)
                print(root)

                shutil.copytree(root, dst)