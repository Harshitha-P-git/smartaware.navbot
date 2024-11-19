import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

# Simulated training data with more distance points and corresponding actions
data = np.array([[5], [10], [15], [20], [25], [30], [35], [40], [45], [50], [60], [70]])
labels = ['stop', 'right', 'left', 'forward', 'left', 'forward', 'right', 'forward', 'left', 'right', 'forward', 'stop']

# Train a more complex model with additional hidden layers
model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=2000, random_state=42)
model.fit(data, labels)

# Save the trained model
joblib.dump(model, 'enhanced_ml_model.pkl')
print("Model trained and saved as 'enhanced_ml_model.pkl'.")

# Export decision logic based on the model's behavior
def generate_decision_logic():
    with open('enhanced_decision_logic.txt', 'w') as f:
        for i in range(5, 75, 5):  # Extended range for rule generation
            action = model.predict([[i]])[0]
            f.write(f"if (distance <= {i}) {{ action = \"{action}\"; }}\n")
    print("Decision logic saved in 'enhanced_decision_logic.txt'.")

# Generate decision logic
generate_decision_logic()
