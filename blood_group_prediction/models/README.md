# Models Directory

This directory should contain your pretrained LSTM model for blood group prediction.

## Required Model

Place your trained model file here as:
```
blood_group_lstm_model.h5
```

## Model Specifications

- **Input Shape**: (1, 14, 1) - 1 timestep, 14 features, 1 channel
- **Output Shape**: (4) - 4 blood group classes [A, B, AB, O]
- **Output Activation**: Softmax
- **Format**: TensorFlow/Keras .h5 file

## Training Notes

If you need to train a model:

1. Prepare dataset with 14 blood parameters and corresponding blood groups
2. Use LSTM architecture with appropriate preprocessing
3. Save the trained model as `.h5` format
4. Place in this directory

## Dummy Model

The application includes a fallback dummy model for demonstration purposes when no pretrained model is available.
