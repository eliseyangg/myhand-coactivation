import sys, os
module_path = os.path.abspath(os.path.join('..', 'coactivation'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from coactivation.process import generate_coactivation_dataset_by_gt

if __name__ == "__main__":
    df = generate_coactivation_dataset_by_gt()
    df.to_csv('coactivations_by_gt.csv', index=False)
