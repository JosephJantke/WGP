import os, sys, importlib, inspect

# Show interpreter & CWD
print("Python:", sys.executable)
print("CWD:", os.getcwd())

# Inspect what 'birdnet_analyzer' is
import birdnet_analyzer
print("birdnet_analyzer __file__:", getattr(birdnet_analyzer, "__file__", None))

# Force-load the *submodule* explicitly
bn_embeddings = importlib.import_module("birdnet_analyzer.embeddings")
print("embeddings module file:", getattr(bn_embeddings, "__file__", None))
print("Has main? ", hasattr(bn_embeddings, "main"))






import birdnet_analyzer.embeddings as bn_embeddings

audio_input       = r"D:\PhD\WGP_model\background_recordings\recordings"
embeddings_output = r"D:\PhD\WGP_model\background_recordings\fp_embeddings"
fmin              = "1000"     # Hz
fmax              = "8000"     # Hz
overlap           = "0.0"      # seconds (0.0 = non-overlapping 3s windows)
threads           = "8"        # Ryzen 3600: 4–8 is a good range
batch_size        = "16"

if __name__ == "__main__":
    os.makedirs(embeddings_output, exist_ok=True)

    args = [
        audio_input,
        "--threads", threads,
        "--batch_size", batch_size,
        "--fmin", fmin,
        "--fmax", fmax,
        "--overlap", overlap,
        "--file_output", embeddings_output,   # ← use your output dir
    ]

    print("Python:", sys.executable)
    print("Working dir:", os.getcwd())
    print("Args:", " ".join(args))

    # ✅ Call the module's main entrypoint
    bn_embeddings.main(args)