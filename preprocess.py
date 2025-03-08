import os

import pandas as pd
import json
from datetime import datetime
import constants


# Convert dataset to JSONL format for training
def preprocess_dataset(csv_path, output_path):
    print(f"Started processing dataset at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_csv(csv_path)
    training_data = []

    for _, row in df.iterrows():
        prompt = f"Analyze this resume and extract key details:\n\n{row['resume_text']}"
        response = {
            "summary": f"Skills: {row['skills']}\nExperience: {row['experience']} years\nEducation: {row['education']}",
            "matched_roles": row['job_role'],
        }
        training_data.append({"prompt": prompt, "response": json.dumps(response)})

    # Create the parent directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"Completed processing dataset at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    preprocess_dataset(constants.dataset_file_path, constants.process_file_path)
