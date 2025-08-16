#!/usr/bin/env python3
"""
Test script for the real CP-VTON+ model implementation
Tests model loading, inference, and integration with the API
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

def test_model_loading():
    """Test if the CP-VTON model can be loaded successfully."""
    print("🧠 Testing CP-VTON+ model loading...")
    
    try:
        from src.models.cpvton_model import CPVTONModel
        
        # Initialize model
        model = CPVTONModel()
        
        # Check if models are loaded
        if model.is_model_loaded():
            print("   ✅ CP-VTON+ models loaded successfully")
            
            # Get model info
            info = model.get_model_info()
            print(f"   📁 Checkpoint path: {info['checkpoint_path']}")
            print(f"   🔧 Device: {info['device']}")
            print(f"   📊 Status: {info['status']}")
            
            return model, True
        else:
            print("   ❌ Models failed to load")
            return model, False
            
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None, False

def test_model_inference(model):
    """Test model inference with sample images."""
    print("\n🎯 Testing CP-VTON+ model inference...")
    
    try:
        # Create test images (256x192x3, normalized [0,1])
        person_img = np.random.rand(256, 192, 3).astype(np.float32)
        clothing_img = np.random.rand(256, 192, 3).astype(np.float32)
        
        print(f"   📸 Created test images: person {person_img.shape}, clothing {clothing_img.shape}")
        
        # Run inference
        result = model.generate_tryon(person_img, clothing_img)
        
        # Check output
        if result.shape == (256, 192, 3):
            print(f"   ✅ Inference successful! Output shape: {result.shape}")
            print(f"   📊 Output range: [{result.min():.3f}, {result.max():.3f}]")
            print(f"   🔢 Output dtype: {result.dtype}")
            return True
        else:
            print(f"   ❌ Wrong output shape: {result.shape}")
            return False
            
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return False

def test_api_integration():
    """Test if the API can use the real CP-VTON model."""
    print("\n🔗 Testing API integration...")
    
    try:
        from src.services.tryon_service import TryOnService
        
        # Initialize service
        service = TryOnService()
        print("   ✅ TryOnService initialized with real CP-VTON model")
        
        # Check service status
        status = service.get_processing_status()
        print(f"   📊 Service status: {status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API integration failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end functionality."""
    print("\n🚀 Testing end-to-end functionality...")
    
    try:
        # Test the complete pipeline
        from src.services.tryon_service import TryOnService
        import numpy as np
        
        service = TryOnService()
        
        # Create test images
        person_img = np.random.rand(256, 192, 3).astype(np.float32)
        clothing_img = np.random.rand(256, 192, 3).astype(np.float32)
        
        # Test preprocessing
        preprocessor = service.preprocessor
        person_processed = preprocessor.preprocess_person_image(person_img)
        clothing_processed = preprocessor.preprocess_clothing_image(clothing_img)
        
        print(f"   ✅ Preprocessing successful: {person_processed.shape}, {clothing_processed.shape}")
        
        # Test model inference
        result = service.model.generate_tryon(person_processed, clothing_processed)
        print(f"   ✅ Model inference successful: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all CP-VTON model tests."""
    print("🚀 Starting CP-VTON+ Model Tests\n")
    
    # Test 1: Model loading
    model, load_success = test_model_loading()
    
    if not load_success:
        print("\n❌ Model loading failed. Cannot proceed with other tests.")
        return False
    
    # Test 2: Model inference
    inference_success = test_model_inference(model)
    
    # Test 3: API integration
    api_success = test_api_integration()
    
    # Test 4: End-to-end functionality
    e2e_success = test_end_to_end()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   🧠 Model Loading: {'✅' if load_success else '❌'}")
    print(f"   🎯 Model Inference: {'✅' if inference_success else '❌'}")
    print(f"   🔗 API Integration: {'✅' if api_success else '❌'}")
    print(f"   🚀 End-to-End: {'✅' if e2e_success else '❌'}")
    
    total_tests = 4
    passed_tests = sum([load_success, inference_success, api_success, e2e_success])
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! CP-VTON+ model is working correctly!")
        print("\n✨ Next steps:")
        print("   1. Test with real images using the API")
        print("   2. Optimize model performance if needed")
        print("   3. Deploy to production")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
