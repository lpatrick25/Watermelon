import requests
import sys
import json
import io
from datetime import datetime
from pathlib import Path

class WatermelonAPITester:
    def __init__(self, base_url="http://localhost:8502/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        # Don't set Content-Type for file uploads
        if not files:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, data=data, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:300]}...")
                self.failed_tests.append({
                    'name': name,
                    'expected': expected_status,
                    'actual': response.status_code,
                    'response': response.text[:300]
                })

            return success, response

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append({
                'name': name,
                'error': str(e)
            })
            return False, None

    def test_health_check(self):
        """Test API health check"""
        return self.run_test("Health Check", "GET", "health", 200)

    def test_api_info(self):
        """Test API info endpoint"""
        return self.run_test("API Info", "GET", "info", 200)

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root Endpoint", "GET", "", 200)

    def test_status_endpoints(self):
        """Test status check endpoints"""
        # Test creating status check
        status_data = {"client_name": "test_client"}
        success, response = self.run_test("Create Status Check", "POST", "status", 200, data=status_data)
        
        # Test getting status checks
        self.run_test("Get Status Checks", "GET", "status", 200)
        
        return success

    def test_predict_without_image(self):
        """Test prediction endpoint without image (should fail)"""
        return self.run_test("Predict Without Image", "POST", "predict", 422)

    def test_predict_with_invalid_file(self):
        """Test prediction endpoint with invalid file"""
        # Create a fake text file
        fake_file = io.BytesIO(b"This is not an image")
        files = {'file': ('test.txt', fake_file, 'text/plain')}
        return self.run_test("Predict With Invalid File", "POST", "predict", 400, files=files)

    def test_list_datasets(self):
        """Test listing datasets"""
        return self.run_test("List Datasets", "GET", "datasets", 200)

    def test_upload_invalid_dataset(self):
        """Test uploading invalid dataset (not a zip file)"""
        fake_file = io.BytesIO(b"This is not a zip file")
        files = {'file': ('test.txt', fake_file, 'text/plain')}
        data = {'dataset_name': 'test_dataset'}
        return self.run_test("Upload Invalid Dataset", "POST", "dataset/upload", 400, data=data, files=files)

    def test_training_status(self):
        """Test training status endpoint"""
        return self.run_test("Training Status", "GET", "train/status", 200)

    def test_start_training_without_dataset(self):
        """Test starting training without valid dataset"""
        training_data = {
            "dataset_name": "nonexistent_dataset",
            "epochs": 5,
            "model_name": "test_model"
        }
        return self.run_test("Start Training (Invalid Dataset)", "POST", "train", 200, data=training_data)

    def test_list_models(self):
        """Test listing models"""
        return self.run_test("List Models", "GET", "models", 200)

    def test_load_nonexistent_model(self):
        """Test loading nonexistent model"""
        return self.run_test("Load Nonexistent Model", "POST", "models/nonexistent_model/load", 404)

    def test_get_predictions(self):
        """Test getting predictions"""
        return self.run_test("Get Predictions", "GET", "predictions", 200)

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Watermelon Classifier API Tests")
        print(f"📍 Base URL: {self.base_url}")
        print("=" * 60)

        # Basic API tests
        self.test_health_check()
        self.test_api_info()
        self.test_root_endpoint()
        
        # Status endpoints
        self.test_status_endpoints()
        
        # Prediction endpoints
        self.test_predict_without_image()
        self.test_predict_with_invalid_file()
        self.test_get_predictions()
        
        # Dataset endpoints
        self.test_list_datasets()
        self.test_upload_invalid_dataset()
        
        # Training endpoints
        self.test_training_status()
        self.test_start_training_without_dataset()
        
        # Model endpoints
        self.test_list_models()
        self.test_load_nonexistent_model()

        # Print final results
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests ({len(self.failed_tests)}):")
            for i, test in enumerate(self.failed_tests, 1):
                print(f"{i}. {test['name']}")
                if 'error' in test:
                    print(f"   Error: {test['error']}")
                else:
                    print(f"   Expected: {test['expected']}, Got: {test['actual']}")
                    print(f"   Response: {test['response']}")
        else:
            print("\n🎉 All tests passed!")

        return self.tests_passed == self.tests_run

def main():
    tester = WatermelonAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())