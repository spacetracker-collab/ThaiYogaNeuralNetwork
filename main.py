import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
NUM_POSTURES = 200
INPUT_DIM = 7   # features per posture
OUTPUT_DIM = 6  # therapeutic characteristics

# -----------------------------
# SYNTHETIC DATA GENERATION
# -----------------------------
def generate_data(n=2000):
    X = []
    Y = []

    for _ in range(n):
        posture_id = np.random.randint(0, NUM_POSTURES)

        # Features
        body_region = posture_id % 5 / 5.0
        stretch = np.random.rand()
        compression = np.random.rand()
        twist = np.random.rand()
        energy_line = np.random.rand()
        rhythm = np.random.rand()
        force = np.random.rand()

        features = [body_region, stretch, compression, twist, energy_line, rhythm, force]

        # Outputs (heuristic mapping)
        relaxation = 0.5 * rhythm + 0.3 * compression
        flexibility = 0.7 * stretch + 0.2 * twist
        circulation = 0.5 * compression + 0.4 * rhythm
        pain_relief = 0.6 * compression + 0.3 * stretch
        nervous_shift = 0.7 * rhythm + 0.2 * energy_line
        energy_balance = 0.8 * energy_line

        outputs = [relaxation, flexibility, circulation,
                   pain_relief, nervous_shift, energy_balance]

        X.append(features)
        Y.append(outputs)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# -----------------------------
# MODEL
# -----------------------------
class ThaiMassageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# TRAINING
# -----------------------------
def train():
    X, Y = generate_data()

    model = ThaiMassageNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, Y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    return model

# -----------------------------
# INFERENCE
# -----------------------------
def predict(model, posture_features):
    with torch.no_grad():
        inp = torch.tensor([posture_features], dtype=torch.float32)
        out = model(inp)[0]
        return {
            "relaxation": float(out[0]),
            "flexibility": float(out[1]),
            "circulation": float(out[2]),
            "pain_relief": float(out[3]),
            "nervous_system": float(out[4]),
            "energy_balance": float(out[5])
        }

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    model = train()

    test_posture = [0.6, 0.8, 0.4, 0.3, 0.7, 0.9, 0.5]
    result = predict(model, test_posture)

    print("\nInference:")
    for k, v in result.items():
        print(k, round(v, 3))
