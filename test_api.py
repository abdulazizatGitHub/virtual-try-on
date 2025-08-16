#!/usr/bin/env python3
"""
Test script for the Virtual Try-On API
Tests all endpoints to ensure they're working correctly
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Root endpoint working: {data['message']}")
            print(f"   📋 Available endpoints: {data['endpoints']}")
            return True
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Root endpoint error: {e}")
        return False

def test_status_endpoint():
    """Test the status endpoint."""
    print("\n📊 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/status")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                status_data = data['data']
                print(f"   ✅ Status endpoint working: {status_data['model']}")
                print(f"   🎯 Target resolution: {status_data['target_resolution']}")
                return True
            else:
                print(f"   ❌ Status endpoint failed: {data}")
                return False
        else:
            print(f"   ❌ Status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status endpoint error: {e}")
        return False

def test_tryon_endpoint_no_files():
    """Test the try-on endpoint without files (should fail gracefully)."""
    print("\n📤 Testing try-on endpoint (no files)...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/tryon")
        if response.status_code == 400:
            data = response.json()
            if not data.get('success') and 'Missing person_image file' in data.get('error', {}).get('message', ''):
                print("   ✅ Try-on endpoint correctly rejected request without files")
                return True
            else:
                print(f"   ❌ Unexpected response: {data}")
                return False
        else:
            print(f"   ❌ Expected 400 error, got: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Try-on endpoint error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Starting Virtual Try-On API Tests\n")
    
    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_status_endpoint,
        test_tryon_endpoint_no_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        print("\n✨ Next steps:")
        print("   1. Start the Flask server: python app.py")
        print("   2. Test with actual image files")
        print("   3. Replace placeholder CP-VTON model with real implementation")
    else:
        print("❌ Some tests failed. Check the API implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
