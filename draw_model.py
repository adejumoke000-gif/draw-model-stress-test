import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------- Derived Weights from Backtesting ----------
LAYER_WEIGHTS = {
    1: 15,  # Strength Balance
    2: 20,  # Expected Goals
    3: 8,   # Tempo
    4: 12,  # League Draw Bias
    5: 10,  # Motivation Symmetry
    6: 25,  # Historical Pattern (H2H)
    7: 10   # Correct Score Band
}

def sigmoid_boost(aligned_layers, max_boost=20):
    return max_boost / (1 + np.exp(-0.5 * (aligned_layers - 3.5)))

def draw_model(match, simulate_uncertainty=True):
    strength_diff = abs(match["home_strength"] - match["away_strength"])
    layer1 = strength_diff <= 0.12
    combined_lambda = match["home_xg"] + match["away_xg"]
    layer2 = 1.3 <= combined_lambda <= 2.3
    layer3 = match["tempo"] <= 0.4
    layer4 = match["league_draw_rate"] >= 0.25
    layer5 = match["motivation_balance"] >= 0.8
    layer6 = match["h2h_draw_rate"] >= 0.25
    layer7 = abs(match.get("home_goals", 0) - match.get("away_goals", 0)) <= 1
    layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
    aligned_layers = sum(layers)

    late_risk_score = (match["home_late_goal_pct"] + match["away_late_goal_pct"]) / 2
    if late_risk_score >= 0.35:
        return {"status": "AVOID", "reason": f"Late goal risk {late_risk_score:.2f}", "confidence": 0}

    weighted_sum = sum(LAYER_WEIGHTS[i+1] for i, aligned in enumerate(layers) if aligned)
    boost = sigmoid_boost(aligned_layers)
    score = weighted_sum + boost

    base_confidence = 70 + (aligned_layers * 5) + (score / 10)
    if simulate_uncertainty:
        noises = np.random.normal(0, 5, 1000)
        confidences = [min(base_confidence + noise, 100) for noise in noises]
        confidence = np.mean(confidences)
        confidence_ci = (np.percentile(confidences, 5), np.percentile(confidences, 95))
    else:
        confidence = min(base_confidence, 100)
        confidence_ci = None

    status = "PLAY DRAW" if confidence >= 80 and aligned_layers >= 5 else "AVOID"

    return {
        "pick": "DRAW",
        "combined_lambda": round(combined_lambda, 2),
        "aligned_layers": aligned_layers,
        "custom_score": round(score, 2),
        "confidence": round(confidence, 2),
        "confidence_ci": confidence_ci,
        "status": status
    }

def backtest_model(matches_data, model_func=draw_model):
    predictions = []
    for match in matches_data:
        result = model_func(match, simulate_uncertainty=False)
        pred_draw = result["status"] == "PLAY DRAW"
        actual_draw = match["actual_draw"]
        predictions.append((pred_draw, actual_draw))
    
    y_pred, y_true = zip(*predictions)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall}
