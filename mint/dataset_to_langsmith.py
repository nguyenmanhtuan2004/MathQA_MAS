import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mint.config import DATA_DIR
from mint.testing.preprocess_data import prepare_qa_input_with_answer_filter, standardize_item

class DatasetLoad:
    def __init__(self):
        load_dotenv()
        self.gsm8k = []
        self.tatqa = []
        self.tabmwp = []
        self._load_datasets()

    def _load_datasets(self):
        """Load all datasets from files."""
        # Load GSM8K dataset
        file_path = os.path.join(DATA_DIR('FILTER_DATASET'), 'gsm8k.jsonl')
        with open(file_path, "r", encoding="utf-8") as f:
            self.gsm8k = [json.loads(line) for line in f]
        
        # Load TABMWP dataset
        folder_path = DATA_DIR('FILTER_DATASET')
        filename = "tabmwp.json"
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            tabmwp_pre = json.load(f)
        
        self.tabmwp = []
        for item in tabmwp_pre:
            self.tabmwp.extend(standardize_item(item, "tabmwp"))
        
        # Load TATQA dataset
        folder_path = DATA_DIR('FILTER_DATASET')
        filename = "tatqa.json"
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data_tatqa = json.load(f)
        tatqa_pre = prepare_qa_input_with_answer_filter(raw_data_tatqa)
        self.tatqa = []
        for item in tatqa_pre:
            self.tatqa.extend(standardize_item(item, "tatqa"))
    
    def get_dataset(self, dataset_name: str):
        """
        Get dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset ('gsm8k', 'tatqa', 'tabmwp')
            
        Returns:
            list: The requested dataset
        """
        dataset_name = dataset_name.lower()
        if dataset_name == 'gsm8k':
            return self.gsm8k
        elif dataset_name == 'tatqa':
            return self.tatqa
        elif dataset_name == 'tabmwp':
            return self.tabmwp
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: gsm8k, tatqa, tabmwp")
    
    def get_all_datasets(self):
        """
        Get all datasets as a dictionary.
        
        Returns:
            dict: Dictionary containing all datasets
        """
        return {
            'gsm8k': self.gsm8k,
            'tatqa': self.tatqa,
            'tabmwp': self.tabmwp
        }

class DatasetToLangsmith:
    def __init__(self, limit: int):
        """
        Initialize the class with a limit for the number of samples to test.
        
        Args:
            limit (int): Number of samples to use for testing
        """
        self.limit = limit
        load_dotenv()
        self.client = Client()
        
        # Load datasets using DatasetLoad class
        dataset_loader = DatasetLoad()
        self.gsm8k = dataset_loader.gsm8k
        self.tatqa = dataset_loader.tatqa
        self.tabmwp = dataset_loader.tabmwp
    
    def _get_or_create_dataset(self, dataset_name: str):
        """
        Get existing dataset or create a new one with timestamp if it already exists.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dataset object
        """
        try:
            # Try to create dataset with original name
            dataset = self.client.create_dataset(dataset_name=dataset_name)
            return dataset
        except Exception as e:
            if "already exists" in str(e):
                # If dataset exists, create a new one with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{dataset_name}_{timestamp}"
                print(f"⚠️  Dataset '{dataset_name}' đã tồn tại. Tạo dataset mới với tên: '{new_name}'")
                dataset = self.client.create_dataset(dataset_name=new_name)
                return dataset
            else:
                raise e
    
    
    def create_gsm8k_dataset(self):
        """Create GSM8K dataset in LangSmith."""
        test_samples = self.gsm8k[:self.limit]
        
        dataset_gsm8k = self._get_or_create_dataset("GSM8K")
        inputs = [{"question": item["question"]} for item in test_samples]
        outputs = [{"answer": item["answer"]} for item in test_samples]
        
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_gsm8k.id,
        )
        return dataset_gsm8k
    
    def create_tatqa_dataset(self):
        """Create TATQA dataset in LangSmith."""
        test_samples = self.tatqa[:self.limit]
        
        dataset_tatqa = self._get_or_create_dataset("TATQA")
        
        inputs = [{"question": item["question"], "context": item["context"]} for item in test_samples]
        outputs = [{"answer": item["answer"]} for item in test_samples]
        
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_tatqa.id,
        )
        return dataset_tatqa
    
    def create_tabmwp_dataset(self):
        """Create TABMWP dataset in LangSmith."""
        test_samples = self.tabmwp[:self.limit]
        
        dataset_tabmwp = self._get_or_create_dataset("TABMWP")
        
        inputs = [{"question": item["question"], "context": item["context"]} for item in test_samples]
        outputs = [{"answer": item["answer"]} for item in test_samples]
        
        self.client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_tabmwp.id,
        )
        return dataset_tabmwp
    
    def create_all_datasets(self):
        """Create all datasets in LangSmith."""
        gsm8k_dataset = self.create_gsm8k_dataset()
        tatqa_dataset = self.create_tatqa_dataset()
        tabmwp_dataset = self.create_tabmwp_dataset()
        
        return {
            "gsm8k": gsm8k_dataset,
            "tatqa": tatqa_dataset,
            "tabmwp": tabmwp_dataset
        }