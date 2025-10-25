import os
import math
import json
import numpy as np
import pandas as pd

try:
    from tqdm import trange
    _use_tqdm = True
except Exception:
    _use_tqdm = False

RNG = np.random.default_rng(42)

# -------------------------------
# Utility: Standard Scaler
# -------------------------------
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_  = X.std(axis=0, ddof=0, keepdims=True)
        # avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -------------------------------
# Model: Simple MLP for binary classification
# -------------------------------
class MLPBinary:
    """
    MLP with L layers:
      input -> [Dense(sigmoid)] x (L-1 hidden layers) -> Dense(sigmoid output)
    Loss: Binary cross-entropy
    Optim: SGD with momentum (optional); L2 (weight decay) optional
    """
    def __init__(self, layer_sizes, lr=0.05, momentum=0.9, weight_decay=0.0, seed=42):
        assert len(layer_sizes) >= 2, "Need input and output sizes"
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)

        # Initialize parameters (Xavier for sigmoid)
        self.W = []
        self.b = []
        self.vW = []
        self.vb = []
        for i in range(len(layer_sizes)-1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            W_i = self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
            b_i = np.zeros((1, fan_out), dtype=np.float64)
            self.W.append(W_i)
            self.b.append(b_i)
            self.vW.append(np.zeros_like(W_i))
            self.vb.append(np.zeros_like(b_i))

    @staticmethod
    def sigmoid(z):
        # stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(a):
        # derivative wrt z, but given a = sigmoid(z): a*(1-a)
        return a * (1.0 - a)

    @staticmethod
    def bce_loss(y_true, y_prob, eps=1e-10):
        # y_true, y_prob shape (N,1)
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    def forward(self, X):
        """Return activations for all layers (including input) and pre-activations z for hidden+output"""
        a = X
        activations = [a]
        zs = []
        for i in range(len(self.W)-1):
            z = a @ self.W[i] + self.b[i]
            a = self.sigmoid(z)
            zs.append(z)
            activations.append(a)
        # output layer
        z = a @ self.W[-1] + self.b[-1]
        a = self.sigmoid(z)
        zs.append(z)
        activations.append(a)
        return activations, zs

    def backward(self, activations, zs, y_true):
        """
        Compute gradients via backpropagation.
        y_true shape: (N,1)
        activations: [a0 (X), a1, ..., aL]
        zs: [z1, ..., zL]
        """
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        aL = activations[-1]
        # dL/daL = -(y/a - (1-y)/(1-a))
        # For sigmoid + BCE, dL/dzL = aL - y
        delta = (aL - y_true)  # shape (N,1)

        # output layer grads
        a_prev = activations[-2]
        grads_W[-1] = (a_prev.T @ delta) / y_true.shape[0]
        grads_b[-1] = np.mean(delta, axis=0, keepdims=True)

        # include L2 weight decay
        if self.weight_decay > 0.0:
            grads_W[-1] += self.weight_decay * self.W[-1]

        # hidden layers (reverse)
        for i in range(len(self.W)-2, -1, -1):
            # delta for layer i (using sigmoid derivative)
            da = delta @ self.W[i+1].T
            delta = da * self.sigmoid_deriv(activations[i+1])
            grads_W[i] = (activations[i].T @ delta) / y_true.shape[0]
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)
            if self.weight_decay > 0.0:
                grads_W[i] += self.weight_decay * self.W[i]

        return grads_W, grads_b

    def step(self, grads_W, grads_b):
        # SGD + momentum
        for i in range(len(self.W)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    def predict_proba(self, X):
        a, _ = self.forward(X)
        return a[-1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def fit(self, X, y, X_val=None, y_val=None, epochs=500, batch_size=32, print_every=50):
        n = X.shape[0]
        history = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "val_acc": []}
        idx = np.arange(n)

        if _use_tqdm:
            it = trange(1, epochs+1)
        else:
            it = range(1, epochs+1)

        for epoch in it:
            # shuffle
            RNG.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # mini-batches
            for start in range(0, n, batch_size):
                end = start + batch_size
                xb = X[start:end]
                yb = y[start:end]
                activations, zs = self.forward(xb)
                grads_W, grads_b = self.backward(activations, zs, yb)
                self.step(grads_W, grads_b)

            # metrics
            a_train = self.predict_proba(X)
            loss = self.bce_loss(y, a_train)
            acc = (self.predict(X) == y).mean()

            if X_val is not None and y_val is not None:
                a_val = self.predict_proba(X_val)
                val_loss = self.bce_loss(y_val, a_val)
                val_acc = (self.predict(X_val) == y_val).mean()
            else:
                val_loss, val_acc = np.nan, np.nan

            history["epoch"].append(epoch)
            history["loss"].append(float(loss))
            history["val_loss"].append(float(val_loss))
            history["acc"].append(float(acc))
            history["val_acc"].append(float(val_acc))

            if not _use_tqdm and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:4d} | loss={loss:.4f} acc={acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if _use_tqdm:
                it.set_description(f"loss {loss:.4f} val_acc {val_acc:.4f}")

        return history


# -------------------------------
# Titanic preprocessing
# -------------------------------
def preprocess_titanic(df: pd.DataFrame, is_train=True):
    df = df.copy()

    # Keep essential columns
    keep = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    if "Survived" not in df.columns:
        keep.remove("Survived")

    df = df[[c for c in keep if c in df.columns]]

    # Sex to 0/1
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype("float64")

    # Embarked fill and one-hot
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        df = pd.concat([df.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    # Age: impute by median grouped on Pclass & Sex if available, else global median
    if "Age" in df.columns:
        if "Pclass" in df.columns and "Sex" in df.columns:
            df["Age"] = df["Age"].astype("float64")
            df["Age"] = df["Age"].fillna(
                df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
            )
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fare: fillna and log1p
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].astype("float64")
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["Fare"] = np.log1p(df["Fare"])

    # SibSp, Parch as-is (fillna 0 just in case)
    for c in ["SibSp", "Parch", "Pclass"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("float64")

    y = None
    if "Survived" in df.columns:
        y = df["Survived"].astype("int64").to_numpy().reshape(-1, 1)
        X = df.drop(columns=["Survived"])
    else:
        X = df

    # Ensure consistent column order (for train/test alignment)
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y


def load_data(data_dir="./titanic", val_ratio=0.2, seed=42):
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Put Kaggle Titanic CSVs in {data_dir}/")

    df_train = pd.read_csv(train_path)
    X_df, y = preprocess_titanic(df_train, is_train=True)

    # train/val split (stratified by Survived)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    # stratify
    pos_idx = idx[y.flatten() == 1]
    neg_idx = idx[y.flatten() == 0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_pos_val = int(len(pos_idx) * val_ratio)
    n_neg_val = int(len(neg_idx) * val_ratio)
    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    X = X_df.to_numpy().astype(np.float64)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # Fit scaler on train only
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # Keep feature names and scaler for later (test)
    feature_names = list(X_df.columns)

    # Load test (optional)
    test_path = os.path.join(data_dir, "test.csv")
    X_test_scaled, test_passenger_ids = None, None
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        # Keep PassengerId for submission
        test_passenger_ids = df_test["PassengerId"].to_numpy() if "PassengerId" in df_test.columns else None

        X_test_df, _ = preprocess_titanic(df_test, is_train=False)
        # Align columns with train
        for col in feature_names:
            if col not in X_test_df.columns:
                X_test_df[col] = 0.0
        X_test_df = X_test_df.reindex(feature_names, axis=1)
        X_test = X_test_df.to_numpy().astype(np.float64)
        X_test_scaled = scaler.transform(X_test)

    return (X_train, y_train, X_val, y_val, scaler, feature_names, X_test_scaled, test_passenger_ids)


def accuracy(y_true, y_pred):
    return (y_true.flatten() == y_pred.flatten()).mean()


def main():
    # Hyperparameters
    H1 = 16
    H2 = 8
    LR = 0.08
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    EPOCHS = 600
    BATCH_SIZE = 32
    VAL_RATIO = 0.2
    SEED = 42

    print("Loading data...")
    try:
        X_train, y_train, X_val, y_val, scaler, feats, X_test, test_ids = load_data("./titanic", val_ratio=VAL_RATIO, seed=SEED)
    except FileNotFoundError as e:
        print(str(e))
        print("Tip: Put Kaggle CSVs at ./titanic/train.csv and ./titanic/test.csv, then rerun.")
        return

    input_dim = X_train.shape[1]
    print(f"Features ({input_dim}): {feats}")

    # Build model
    model = MLPBinary(
        layer_sizes=[input_dim, H1, H2, 1],
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        seed=SEED
    )

    print("Training...")
    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, print_every=50
    )

    # Final metrics
    yhat_train = model.predict(X_train)
    yhat_val   = model.predict(X_val)
    train_acc = accuracy(y_train, yhat_train)
    val_acc   = accuracy(y_val, yhat_val)
    print(f"\nFinal Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save metrics
    metrics = {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "history": history
    }
    os.makedirs("./titanic", exist_ok=True)
    with open("./titanic/nn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Predict test if available
    if X_test is not None and test_ids is not None:
        preds = model.predict(X_test, threshold=0.5).flatten()
        sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
        sub_path = "./titanic/submission_nn.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Submission written to {sub_path}")

if __name__ == "__main__":
    main()
