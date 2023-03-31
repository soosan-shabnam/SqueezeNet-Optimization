OUTPUT_PATH = "output"

INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 10

TRAIN_PATH = '../data/grayscale/train'
TEST_PATH = '../data/grayscale/test'

BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"