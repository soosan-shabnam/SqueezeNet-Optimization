from src import config

from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils.layer_utils import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils


# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand_1=64, expand_3=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + config.sq1x1)(x)
    x = Activation('relu', name=s_id + config.relu + config.sq1x1)(x)

    left = Convolution2D(expand_1, (1, 1), padding='valid', name=s_id + config.exp1x1)(x)
    left = Activation('relu', name=s_id + config.relu + config.exp1x1)(left)

    right = Convolution2D(expand_3, (3, 3), padding='same', name=s_id + config.exp3x3)(x)
    right = Activation('relu', name=s_id + config.relu + config.exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')

    return x


# Original SqueezeNet from paper.
def SqueezeNet(hp, include_top=True, weights='imagenet',
               input_tensor=None, input_shape=config.INPUT_SHAPE,
               pooling=None,
               classes=1000):
    """Instantiates the SqueezeNet architecture.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=hp.Int("fire2_squeeze", min_value=16, max_value=64, step=16),
                    expand_1=hp.Int("fire2_expand1", min_value=64, max_value=256, step=16),
                    expand_3=hp.Int("fire2_expand3", min_value=64, max_value=256, step=16))
    x = fire_module(x, fire_id=3, squeeze=hp.Int("fire3_squeeze", min_value=16, max_value=64, step=16),
                    expand_1=hp.Int("fire3_expand1", min_value=64, max_value=256, step=16),
                    expand_3=hp.Int("fire3_expand3", min_value=64, max_value=256, step=16))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=hp.Int("fire4_squeeze", min_value=32, max_value=128, step=16),
                    expand_1=hp.Int("fire4_expand1", min_value=128, max_value=512, step=16),
                    expand_3=hp.Int("fire4_expand3", min_value=128, max_value=512, step=16))
    x = fire_module(x, fire_id=5, squeeze=hp.Int("fire5_squeeze", min_value=32, max_value=128, step=16),
                    expand_1=hp.Int("fire5_expand1", min_value=128, max_value=512, step=16),
                    expand_3=hp.Int("fire5_expand3", min_value=128, max_value=512, step=16))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=hp.Int("fire6_squeeze", min_value=48, max_value=192, step=16),
                    expand_1=hp.Int("fire6_expand1", min_value=192, max_value=768, step=16),
                    expand_3=hp.Int("fire6_expand3", min_value=192, max_value=768, step=16))
    x = fire_module(x, fire_id=7, squeeze=hp.Int("fire7_squeeze", min_value=48, max_value=192, step=16),
                    expand_1=hp.Int("fire7_expand1", min_value=192, max_value=768, step=16),
                    expand_3=hp.Int("fire7_expand3", min_value=192, max_value=768, step=16))
    x = fire_module(x, fire_id=8, squeeze=hp.Int("fire8_squeeze", min_value=64, max_value=256, step=16),
                    expand_1=hp.Int("fire8_expand1", min_value=256, max_value=1024, step=16),
                    expand_3=hp.Int("fire8_expand3", min_value=256, max_value=1024, step=16))
    x = fire_module(x, fire_id=9, squeeze=hp.Int("fire9_squeeze", min_value=64, max_value=256, step=16),
                    expand_1=hp.Int("fire9_expand1", min_value=256, max_value=1024, step=16),
                    expand_3=hp.Int("fire9_expand3", min_value=256, max_value=1024, step=16))

    if include_top:
        # It's not obvious where to cut the network...
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
        x = Dropout(0.5, name='drop9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = Activation('relu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        elif pooling is None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    config.WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    config.WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                print('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
    return model
