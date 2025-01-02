# SnackTrack

Prototype implementation of a deep learning approach for audio based food intake detection using a throat microphone.

## Project Structure

- [database_creation](./database_creation) - Contains the scripts used for creating the dataset.
- [SnackTrackBasicModel.py](./SnackTrackBasicModel.py) - Definition of a basic CNN Model trained for detecting food intakes.
- [SnackTrackDataset.py](./SnackTrackDataset.py) - Class definition to use the dataset for training with the pytorch package.
- [CNN_Training.ipynb](./CNN_Training.ipynb) - Script for training the model.
- [detectSnackEvent.ipynb](./detectSnackEvent.ipynb) - Sample implementation of how to use the trained model on continous audio signal.