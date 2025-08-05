from langsmith import Client
import pandas as pd
import json
import os


class ExperimentDataFetcher:
    """
    Class to fetch experiment data from LangSmith.
    """
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id

    def fetch_data(self):
        client = Client()
        runs = list(client.list_runs(project_name=self.experiment_id, execution_order=1))

        data = []
        count = 0
        for run in runs:
            is_correct = None
            feedback_list = client.list_feedback(run_ids=[run.id])
            for fb in feedback_list:
                if fb.key == "is_correct":
                    is_correct = fb.score
            count+=1
            row = {
                "run_id": run.id,
                "error": run.error,
                "latency_sec": (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None,
                "total_cost": run.total_cost,
                "input_tokens": run.prompt_tokens,
                "output_tokens": run.completion_tokens,
                "total_tokens": run.total_tokens,
                "is_correct": is_correct,
            }
            data.append(row)

        df = pd.DataFrame(data)
        parts = self.experiment_id.split(" - ")
        dataset = parts[0]                     
        subparts = parts[1].split("-")        
        method = subparts[0]   
        
        # Generate filename based on experiment_id
        filename = f"{method}_{dataset}.json"

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Full path for the JSON file
        filepath = os.path.join("results", filename)
        
        # Convert DataFrame to JSON and save
        df.to_json(filepath, orient='records', indent=2)
        print(f"Data saved to: {filepath}")
        
class ChartGenerate:
    """
    Class to generate charts from experiment data.
    """
    def __init__(self, methods, datasets, metric):
        self.methods = methods
        self.datasets = datasets
        self.metric = metric

    def load_data(self, method, dataset, folder="results"):
        file_path = os.path.join(folder, f"{method}_{dataset}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        return pd.read_json(file_path)

    def generate_chart(self):
        """
        Generate charts from loaded data.
        """
        pass