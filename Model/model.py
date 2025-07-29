import tensorflow as tf
from keras.layers import TFSMLayer
from tensorflow.keras import Input, Model

### Preparing BirdNET 2.4 as a callable layer for the CNN-backbone of the model:

# Define input shape (BirdNET expects RGB Mel spectrograms)
low_input = Input(shape=(96, 512, 1), name="low_band")         #todo check these input parameters
high_input = Input(shape=(96, 512, 1), name="high_band")

# Load the 2.4 model as a callable frozen layer
birdnet_layer = TFSMLayer(
    "V2.4/BirdNET_GLOBAL_6K_V2.4_Model",
    call_endpoint="serving_default"
)

# Use BirdNET as a frozen feature extractor
birdnet_output = birdnet_layer({"low": low_input, "high": high_input})


#todo  You can now add a transformer head or classifier here?
model = Model(inputs=[low_input, high_input], outputs=birdnet_output)

#model. summary() provides a summary of the model with output shapes and number of parameters at each step

#todo work out whether you need to remove the classifier head or just ignore it


### Specifying the model in TensorFlow. Using pretrained Birdnet 2.3 as CNN-bakcbone

