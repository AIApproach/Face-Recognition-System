import tensorflow as tf
from tensorflow.keras import layers, Model

# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.name_scope(scope or 'Block35'):
        with tf.name_scope('Branch_0'):
            tower_conv = layers.Conv2D(32, 1, padding='same', use_bias=False)(net)
            tower_conv = layers.BatchNormalization()(tower_conv)
            tower_conv = layers.Activation('relu')(tower_conv)
            
        with tf.name_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(32, 1, padding='same', use_bias=False)(net)
            tower_conv1_0 = layers.BatchNormalization()(tower_conv1_0)
            tower_conv1_0 = layers.Activation('relu')(tower_conv1_0)
            
            tower_conv1_1 = layers.Conv2D(32, 3, padding='same', use_bias=False)(tower_conv1_0)
            tower_conv1_1 = layers.BatchNormalization()(tower_conv1_1)
            tower_conv1_1 = layers.Activation('relu')(tower_conv1_1)
            
        with tf.name_scope('Branch_2'):
            tower_conv2_0 = layers.Conv2D(32, 1, padding='same', use_bias=False)(net)
            tower_conv2_0 = layers.BatchNormalization()(tower_conv2_0)
            tower_conv2_0 = layers.Activation('relu')(tower_conv2_0)
            
            tower_conv2_1 = layers.Conv2D(48, 3, padding='same', use_bias=False)(tower_conv2_0)
            tower_conv2_1 = layers.BatchNormalization()(tower_conv2_1)
            tower_conv2_1 = layers.Activation('relu')(tower_conv2_1)
            
            tower_conv2_2 = layers.Conv2D(64, 3, padding='same', use_bias=False)(tower_conv2_1)
            tower_conv2_2 = layers.BatchNormalization()(tower_conv2_2)
            tower_conv2_2 = layers.Activation('relu')(tower_conv2_2)
            
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = layers.Conv2D(net.shape[-1], 1, padding='same', use_bias=True)(mixed)
        
        net = net + scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.name_scope(scope or 'Block17'):
        with tf.name_scope('Branch_0'):
            tower_conv = layers.Conv2D(192, 1, padding='same', use_bias=False)(net)
            tower_conv = layers.BatchNormalization()(tower_conv)
            tower_conv = layers.Activation('relu')(tower_conv)
            
        with tf.name_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(128, 1, padding='same', use_bias=False)(net)
            tower_conv1_0 = layers.BatchNormalization()(tower_conv1_0)
            tower_conv1_0 = layers.Activation('relu')(tower_conv1_0)
            
            tower_conv1_1 = layers.Conv2D(160, [1, 7], padding='same', use_bias=False)(tower_conv1_0)
            tower_conv1_1 = layers.BatchNormalization()(tower_conv1_1)
            tower_conv1_1 = layers.Activation('relu')(tower_conv1_1)
            
            tower_conv1_2 = layers.Conv2D(192, [7, 1], padding='same', use_bias=False)(tower_conv1_1)
            tower_conv1_2 = layers.BatchNormalization()(tower_conv1_2)
            tower_conv1_2 = layers.Activation('relu')(tower_conv1_2)
            
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = layers.Conv2D(net.shape[-1], 1, padding='same', use_bias=True)(mixed)
        
        net = net + scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.name_scope(scope or 'Block8'):
        with tf.name_scope('Branch_0'):
            tower_conv = layers.Conv2D(192, 1, padding='same', use_bias=False)(net)
            tower_conv = layers.BatchNormalization()(tower_conv)
            tower_conv = layers.Activation('relu')(tower_conv)
            
        with tf.name_scope('Branch_1'):
            tower_conv1_0 = layers.Conv2D(192, 1, padding='same', use_bias=False)(net)
            tower_conv1_0 = layers.BatchNormalization()(tower_conv1_0)
            tower_conv1_0 = layers.Activation('relu')(tower_conv1_0)
            
            tower_conv1_1 = layers.Conv2D(224, [1, 3], padding='same', use_bias=False)(tower_conv1_0)
            tower_conv1_1 = layers.BatchNormalization()(tower_conv1_1)
            tower_conv1_1 = layers.Activation('relu')(tower_conv1_1)
            
            tower_conv1_2 = layers.Conv2D(256, [3, 1], padding='same', use_bias=False)(tower_conv1_1)
            tower_conv1_2 = layers.BatchNormalization()(tower_conv1_2)
            tower_conv1_2 = layers.Activation('relu')(tower_conv1_2)
            
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = layers.Conv2D(net.shape[-1], 1, padding='same', use_bias=True)(mixed)
        
        net = net + scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

def inception_resnet_v2(inputs, is_training=True,
                      dropout_keep_prob=0.8,
                      bottleneck_layer_size=128,
                      reuse=None,
                      scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model."""
    end_points = {}
    
    # Input layer
    x = inputs
    
    # Stem block
    with tf.name_scope(scope):
        # 149 x 149 x 32
        x = layers.Conv2D(32, 3, strides=2, padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_1a_3x3'] = x
        
        # 147 x 147 x 32
        x = layers.Conv2D(32, 3, padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_2a_3x3'] = x
        
        # 147 x 147 x 64
        x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_2b_3x3'] = x
        
        # 73 x 73 x 64
        x = layers.MaxPool2D(3, strides=2, padding='valid')(x)
        end_points['MaxPool_3a_3x3'] = x
        
        # 73 x 73 x 80
        x = layers.Conv2D(80, 1, padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_3b_1x1'] = x
        
        # 71 x 71 x 192
        x = layers.Conv2D(192, 3, padding='valid', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_4a_3x3'] = x
        
        # 35 x 35 x 192
        x = layers.MaxPool2D(3, strides=2, padding='valid')(x)
        end_points['MaxPool_5a_3x3'] = x
        
        # Mixed 5b (Inception-A block)
        with tf.name_scope('Mixed_5b'):
            branch0 = layers.Conv2D(96, 1, padding='same', use_bias=False)(x)
            branch0 = layers.BatchNormalization()(branch0, training=is_training)
            branch0 = layers.Activation('relu')(branch0)
            
            branch1 = layers.Conv2D(48, 1, padding='same', use_bias=False)(x)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            branch1 = layers.Conv2D(64, 5, padding='same', use_bias=False)(branch1)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            
            branch2 = layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            branch2 = layers.Conv2D(96, 3, padding='same', use_bias=False)(branch2)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            branch2 = layers.Conv2D(96, 3, padding='same', use_bias=False)(branch2)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            
            branch3 = layers.AveragePooling2D(3, strides=1, padding='same')(x)
            branch3 = layers.Conv2D(64, 1, padding='same', use_bias=False)(branch3)
            branch3 = layers.BatchNormalization()(branch3, training=is_training)
            branch3 = layers.Activation('relu')(branch3)
            
            x = tf.concat([branch0, branch1, branch2, branch3], axis=-1)
        
        end_points['Mixed_5b'] = x
        
        # 10x block35 (Inception-ResNet-A block)
        for i in range(10):
            x = block35(x, scale=0.17, activation_fn=tf.nn.relu, scope=f'Block35_{i}')
        
        # Mixed 6a (Reduction-A block)
        with tf.name_scope('Mixed_6a'):
            branch0 = layers.Conv2D(384, 3, strides=2, padding='valid', use_bias=False)(x)
            branch0 = layers.BatchNormalization()(branch0, training=is_training)
            branch0 = layers.Activation('relu')(branch0)
            
            branch1 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            branch1 = layers.Conv2D(256, 3, padding='same', use_bias=False)(branch1)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            branch1 = layers.Conv2D(384, 3, strides=2, padding='valid', use_bias=False)(branch1)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            
            branch2 = layers.MaxPool2D(3, strides=2, padding='valid')(x)
            
            x = tf.concat([branch0, branch1, branch2], axis=-1)
        
        end_points['Mixed_6a'] = x
        
        # 20x block17 (Inception-ResNet-B block)
        for i in range(20):
            x = block17(x, scale=0.10, activation_fn=tf.nn.relu, scope=f'Block17_{i}')
        
        # Mixed 7a (Reduction-B block)
        with tf.name_scope('Mixed_7a'):
            branch0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
            branch0 = layers.BatchNormalization()(branch0, training=is_training)
            branch0 = layers.Activation('relu')(branch0)
            branch0 = layers.Conv2D(384, 3, strides=2, padding='valid', use_bias=False)(branch0)
            branch0 = layers.BatchNormalization()(branch0, training=is_training)
            branch0 = layers.Activation('relu')(branch0)
            
            branch1 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            branch1 = layers.Conv2D(288, 3, strides=2, padding='valid', use_bias=False)(branch1)
            branch1 = layers.BatchNormalization()(branch1, training=is_training)
            branch1 = layers.Activation('relu')(branch1)
            
            branch2 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            branch2 = layers.Conv2D(288, 3, padding='same', use_bias=False)(branch2)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            branch2 = layers.Conv2D(320, 3, strides=2, padding='valid', use_bias=False)(branch2)
            branch2 = layers.BatchNormalization()(branch2, training=is_training)
            branch2 = layers.Activation('relu')(branch2)
            
            branch3 = layers.MaxPool2D(3, strides=2, padding='valid')(x)
            
            x = tf.concat([branch0, branch1, branch2, branch3], axis=-1)
        
        end_points['Mixed_7a'] = x
        
        # 10x block8 (Inception-ResNet-C block)
        for i in range(9):
            x = block8(x, scale=0.20, activation_fn=tf.nn.relu, scope=f'Block8_{i}')
        x = block8(x, scale=0.20, activation_fn=None, scope='Block8_9')
        
        # Final convolution
        x = layers.Conv2D(1536, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x, training=is_training)
        x = layers.Activation('relu')(x)
        end_points['Conv2d_7b_1x1'] = x
        
        # Logits
        with tf.name_scope('Logits'):
            end_points['PrePool'] = x
            x = layers.GlobalAveragePooling2D()(x)
            end_points['PreLogitsFlatten'] = x
            
            if is_training:
                x = layers.Dropout(1.0 - dropout_keep_prob)(x)
            
            x = layers.Dense(bottleneck_layer_size, name='Bottleneck')(x)
    
    return x, end_points

def inference(images, keep_probability, phase_train=True, 
             bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    """Builds the Inception ResNet v2 model for inference."""
    # Set weight decay
    if weight_decay > 0:
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
    
    # Build model
    logits, end_points = inception_resnet_v2(
        images,
        is_training=phase_train,
        dropout_keep_prob=keep_probability,
        bottleneck_layer_size=bottleneck_layer_size,
        reuse=reuse
    )
    
    return logits, end_points