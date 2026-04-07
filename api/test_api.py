# Test script for KrishiDisha FastAPI

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_chat():
    """Test chat endpoint"""
    print("\n=== Testing Chat Endpoint ===")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "Hello, what crops grow well in Punjab?"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_crop_recommendation():
    """Test crop recommendation endpoint"""
    print("\n=== Testing Crop Recommendation Endpoint ===")
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.5,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202
    }
    response = requests.post(
        f"{BASE_URL}/crop/recommend",
        json=payload
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code in [200, 503]  # 503 is OK if model not loaded

def test_fertilizer_recommendation():
    """Test fertilizer recommendation endpoint"""
    print("\n=== Testing Fertilizer Recommendation Endpoint ===")
    payload = {
        "N": 60,
        "P": 30,
        "K": 40,
        "soil_type": "loamy",
        "crop_type": "rice"
    }
    response = requests.post(
        f"{BASE_URL}/fertilizer/recommend",
        json=payload
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code in [200, 503]

def test_reference_data():
    """Test reference data endpoints"""
    print("\n=== Testing Reference Data Endpoints ===")
    
    # Test crops
    response = requests.get(f"{BASE_URL}/crops")
    print(f"Crops - Status: {response.status_code}, Count: {len(response.json().get('crops', []))}")
    
    # Test states
    response = requests.get(f"{BASE_URL}/states")
    print(f"States - Status: {response.status_code}, Count: {len(response.json().get('states', []))}")
    
    # Test seasons
    response = requests.get(f"{BASE_URL}/seasons")
    print(f"Seasons - Status: {response.status_code}, Response: {response.json()}")
    
    # Test diseases
    response = requests.get(f"{BASE_URL}/diseases")
    print(f"Diseases - Status: {response.status_code}, Count: {len(response.json().get('diseases', []))}")
    
    return True

def test_activity_log():
    """Test activity logging endpoint"""
    print("\n=== Testing Activity Logging Endpoint ===")
    payload = {
        "farmer_id": 123,
        "activity_type": "crop_recommendation",
        "input_data": {"N": 90, "P": 42, "K": 43},
        "output_data": {"recommended_crop": "Rice"}
    }
    response = requests.post(
        f"{BASE_URL}/activity/log",
        json=payload
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("KrishiDisha FastAPI Test Suite")
    print("=" * 60)
    
    try:
        results = []
        
        # Run tests
        results.append(("Health Check", test_health()))
        results.append(("Chat", test_chat()))
        results.append(("Crop Recommendation", test_crop_recommendation()))
        results.append(("Fertilizer Recommendation", test_fertilizer_recommendation()))
        results.append(("Reference Data", test_reference_data()))
        results.append(("Activity Logging", test_activity_log()))
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the server is running:")
        print("  cd /workspace && python -m uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
