
from typing import Optional
from .CoT import CoTTesting
from .PoT import PoTTesting  # Temporarily commented out due to import issues
from .Zero_shot import ZeroShotTesting  # Temporarily commented out due to import issues  
from .PaL import PaLTesting  # Temporarily commented out due to import issues
from .MultiAgent import MultiAgentTesting
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset_to_langsmith import DatasetLoad

class TestingDataset():
    def __init__(self, method: str, dataset: str):
        self.method = method
        self.dataset = dataset

    def run(self):
        # Implement the logic for testing the dataset

        if self.method.lower() == "zero_shot":
            # Logic for Zero-shot testing
            print(f"Đang chạy kiểm tra Zero-shot trên {self.dataset} ở Langsmith...")
            print('-' * 50)
            test_runner = ZeroShotTesting()
            test_runner.testing(name=self.dataset)
        elif self.method.lower() == "pot":
            # Logic for PoT testing
            print(f"Đang chạy kiểm tra PoT trên {self.dataset} ở Langsmith...")
            print('-' * 50)
            test_runner = PoTTesting()
            test_runner.testing(name=self.dataset)
        elif self.method.lower() == "cot":
            # Logic for CoT testing
            print(f"Đang chạy kiểm tra CoT trên {self.dataset} ở Langsmith...")
            print('-' * 50)

            test_runner = CoTTesting()
            test_runner.testing(name=self.dataset)
        elif self.method.lower() == "pal":
            # Logic for PaL testing
            print(f"Đang chạy kiểm tra PaL trên {self.dataset} ở Langsmith...")
            print('-' * 50)

            test_runner = PaLTesting()
            test_runner.testing(name=self.dataset)
        elif self.method.lower() == "multiagent":
            # Logic for MultiAgent testing
            print(f"Đang chạy kiểm tra MultiAgent trên {self.dataset} ở Langsmith...")
            print('-' * 50)

            test_runner = MultiAgentTesting()
            test_runner.testing(name=self.dataset)
        elif self.method.lower() == "all":
            # Logic for testing all methods
            print(f"Đang chạy kiểm tra tất cả phương pháp trên {self.dataset} ở Langsmith...")
            print('-' * 50)
            
            methods = ["zero_shot", "pot", "cot", "pal"]
            for i, method in enumerate(methods, 1):
                print(f"\n🔄 Đang chạy phương pháp {i}/{len(methods)}: {method.upper()}")
                print("=" * 40)
                
                try:
                    if method == "zero_shot":
                        test_runner = ZeroShotTesting()
                        test_runner.testing(name=self.dataset)
                    elif method == "pot":
                        test_runner = PoTTesting()
                        test_runner.testing(name=self.dataset)
                    elif method == "cot":
                        test_runner = CoTTesting()
                        test_runner.testing(name=self.dataset)
                    elif method == "pal":
                        test_runner = PaLTesting()
                        test_runner.testing(name=self.dataset)
                    
                    print(f"✅ Hoàn thành kiểm tra {method.upper()}")
                except Exception as e:
                    print(f"❌ Lỗi khi chạy {method.upper()}: {str(e)}")
                
                print("=" * 40)
            
            print(f"\n🎉 Đã hoàn thành kiểm tra tất cả phương pháp trên {self.dataset}!")
        
        else:
            print(f"❌ Phương pháp '{self.method}' không được hỗ trợ!")
            print("Các phương pháp có sẵn: Zero_shot, PoT, CoT, PaL, all")
        
        # Perform testing using the selected dataset
        pass