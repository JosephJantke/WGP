##prepare the BirdNET CNN
# Define input shape (BirdNET expects RGB Mel spectrograms)
low_input = Input(shape=(48, 431, 1), name="low_band")
high_input = Input(shape=(48, 431, 1), name="high_band")

# Load the 2.4 model as a callable frozen layer
birdnet_layer = TFSMLayer(
    "path/to/BirdNET_GLOBAL_6K_V2.4_Model",
    call_endpoint="serving_default"
)

# Use BirdNET as a frozen feature extractor
birdnet_output = birdnet_layer({"low": low_input, "high": high_input})

#todo  You can now add a transformer head or classifier here? Probably want to do this in model.py
model = Model(inputs=[low_input, high_input], outputs=birdnet_output)

#todo work out whether you need to remove the classifier head or just ignore it