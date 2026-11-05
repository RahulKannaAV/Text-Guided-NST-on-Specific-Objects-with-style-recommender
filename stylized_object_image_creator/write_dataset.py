import pandas as pd
from typing import Dict

def write_new_data(data_list):
    data = pd.DataFrame({
        "content_name": data_list['content'],
        "style_name": data_list['style'],
        "norm_pixel": data_list['pixel'],
        "norm_structural": data_list['structural'],
        "final_score": data_list['final_score']
    })

    data.to_csv("results.csv")

