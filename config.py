
# ========  Training ============
TRAIN_DATA_ROOT = r'J:\research\datasets\GoogleEarth\collection_10000'
# ground truth json file, a dict that key is UAV image name "xxx.bmp" 
# and corresponding value is satellite image name "xxx.jpg"
TRAIN_GT_PATH = r'J:\面试\GatherAlgorithms\deep_learning\seq_models\seq_dis\train_gt.json'


# ========   Verification  ===========
VER_DATA_ROOT = r'J:\research\datasets\GoogleEarth\collection_1'
VER_GT_PATH = r'J:\面试\GatherAlgorithms\deep_learning\seq_models\seq_dis\test_gt.json'

# ===== Hyperparameters ==========
class Hyper():
    def __init__(self):
        self.batchsize = 64

hyper = Hyper()
