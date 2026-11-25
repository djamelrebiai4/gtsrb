# GTSRB Traffic Sign Recognition

Traffic-sign classification for the German Traffic Sign Recognition Benchmark (GTSRB), with a training notebook and a Gradio application.

## Project files

- `gtsrb_resnet_v4.ipynb` trains and evaluates the model.
- `GTSRB_Local/app.py` runs the image-classification application.
- `GTSRB_Local/requirements.txt` lists the application dependencies.
- The PNG files in the repository root are project screenshots.

## Run the application

Install the dependencies from `GTSRB_Local/requirements.txt`. The app expects the model files `gtsrb_resnet.keras` and `best.weights.h5` in `GTSRB_Local/`. Generate them with the training notebook, then place them there before launching:

```bash
cd GTSRB_Local
python app.py
```

The local virtual environment is intentionally excluded from version control. Recreate it and install dependencies on your own machine instead of uploading it.