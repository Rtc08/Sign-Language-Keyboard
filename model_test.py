import keras
model = keras.models.load_model('trained_model_5.h5')
print(f"This model expects {model.input_shape[1]} inputs (Features)")
print(f"This model predicts {model.output_shape[1]} classes (Labels)")