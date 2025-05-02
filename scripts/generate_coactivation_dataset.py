import sys, os
module_path = os.path.abspath(os.path.join('..', 'coactivation'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from coactivation.process import generate_coactivation_dataset

if __name__ == "__main__":
    df = generate_coactivation_dataset()
    df.to_csv('coactivations.csv', index=False)
