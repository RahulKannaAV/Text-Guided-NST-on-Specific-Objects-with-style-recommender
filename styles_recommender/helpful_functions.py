import re
import streamlit as st
import torch
from pathlib import Path
from StyleSelectorModel import StyleSelector

metrics_list = ["Color", "Structure", "Semantic"]

def find_evaluation_metric(user_prompt):

    for metric in metrics_list:
        pattern = re.compile(metric, re.IGNORECASE)

        if(pattern.search(user_prompt)):
            return metric

def recommend_label(metric, content="Squirrel"):
    

    model = StyleSelector(
    num_contents=3,
    num_metrics=3,
    num_styles=2,
    )

    MODEL_PATH = Path("models")
    WEIGHTS_NAME = "rec_model_weights.pth"
    if(not MODEL_PATH.is_dir()):
        print("No model weights found")
        return 
    
    model.load_state_dict(torch.load(MODEL_PATH / WEIGHTS_NAME))
    print("Weights loaded")

    results = predict_style(model, content, metric)

    st.session_state.recommendations = results['predicted_style']


def predict_style(model, content_label, metric_label):

    contents = ['Bear', 'Bird', 'Squirrel']
    styles = ['Sketch', 'Wave']

    content_to_id = {
        name: i for i, name in enumerate(contents)
    }
    style_to_id = {
        name: i for i, name in enumerate(styles)
    }
    metric_to_id = {
        name: i for i, name in enumerate(metrics_list)
    }

    if content_label not in content_to_id:
        raise ValueError(
            f"Unknown content: {content_label}. "
            "This ID-based model only supports "
            "contents seen during training."
        )

    if metric_label not in metric_to_id:
        raise ValueError(
            f"Unknown metric: {metric_label}"
        )

    model.eval()

    content_id = torch.tensor(
        [content_to_id[content_label]]
    )
    metric_id = torch.tensor(
        [metric_to_id[metric_label]]
    )

    with torch.no_grad():
        logits = model(content_id, metric_id)
        probabilities = torch.softmax(logits, dim=1)
        predicted_id = probabilities.argmax(dim=1).item()

    return {
        "content": content_label,
        "metric": metric_label,
        "predicted_style": styles[predicted_id],
        "probabilities": {
            style: round(probabilities[0, i].item(), 4)
            for i, style in enumerate(styles)
        },
    }

