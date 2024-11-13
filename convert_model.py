from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the model with custom objects to ensure 'mse' is recognized
model = load_model("dqn_trading_model.h5", custom_objects={"mse": MeanSquaredError()})

# Save the model in .keras format
model.save("dqn_trading_model.keras")
print("Model has been successfully converted to .keras format.")
