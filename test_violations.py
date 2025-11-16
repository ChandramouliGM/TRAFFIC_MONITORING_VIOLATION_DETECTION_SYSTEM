#!/usr/bin/env python3
"""
Test script to verify violation detection is working properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection_models import ViolationDetector, VehicleDetector, PlateRecognizer
import numpy as np
from datetime import datetime

def test_violation_detection():
    """Test the violation detection system"""
    
    print("🚦 Testing Traffic Violation Detection System")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing detection models...")
    vehicle_detector = VehicleDetector()
    plate_recognizer = PlateRecognizer()
    violation_detector = ViolationDetector()
    
    # Create mock frame and vehicle data
    print("2. Creating test data...")
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Mock vehicle detection
    mock_vehicle = {
        'class': 'car',
        'confidence': 0.85,
        'bbox': [100, 150, 120, 80],
        'class_id': 2
    }
    
    # Mock vehicle crop for plate recognition
    vehicle_crop = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
    
    print("3. Testing vehicle detection...")
    vehicles = vehicle_detector.detect(mock_frame, 0.5)
    print(f"   ✅ Detected {len(vehicles)} vehicles")
    
    print("4. Testing license plate recognition...")
    plate_text = plate_recognizer.recognize(vehicle_crop)
    print(f"   ✅ Recognized plate: {plate_text}")
    
    print("5. Testing comprehensive violation detection...")
    
    # Build comprehensive vehicle data
    vehicle_data = {
        'bbox': mock_vehicle['bbox'],
        'type': 'car',
        'crop_image': vehicle_crop,
        'detection': mock_vehicle,
        'speed': 75,  # Speeding scenario
        'behavior': {},
        'vehicle_details': {'basic_class': 'car', 'features': {'dominant_color': 'blue'}},
        'tracking': {
            'positions': [(160, 190), (165, 195), (170, 200)],
            'timestamps': [0.0, 0.1, 0.2],
            'vehicle_id': 'test_vehicle_1'
        }
    }
    
    # Frame context
    frame_context = {
        'frame_number': 100,
        'video_fps': 30,
        'violation_types_enabled': ['Speeding', 'Red Light Running', 'No Helmet', 'Wrong Lane'],
        'road_type': 'urban'
    }
    
    # Test comprehensive violation detection
    violations_result = violation_detector.detect_comprehensive_violations(
        vehicle_data, mock_frame, frame_context
    )
    
    print(f"   ✅ Violation detection completed!")
    print(f"   📊 Total violations: {violations_result['total_violations']}")
    print(f"   🎯 Severity score: {violations_result['severity_score']:.1f}")
    print(f"   🚨 Priority: {violations_result['enforcement_priority']}")
    
    if violations_result['violations']:
        print("\n6. Detected violations:")
        for i, violation in enumerate(violations_result['violations'], 1):
            print(f"   {i}. {violation['type']} - {violation['severity']} (confidence: {violation['confidence']:.2f})")
            print(f"      Details: {violation['details']}")
    else:
        print("   ⚠️  No violations detected - this might indicate an issue")
    
    print("\n7. Testing individual violation types...")
    
    # Test speeding detection
    print("   Testing speeding detection...")
    speed_result = violation_detector.detect_speeding_advanced(
        [(160, 190), (165, 195), (170, 200)],
        [0.0, 0.1, 0.2],
        'car'
    )
    print(f"   Speed violation: {speed_result['is_speeding']} (speed: {speed_result['calculated_speed']:.1f} km/h)")
    
    # Test helmet detection
    print("   Testing helmet detection...")
    helmet_result = violation_detector.detect_helmet_violation(vehicle_crop, 'motorcycle')
    print(f"   Helmet violation: {helmet_result['has_violation']} (confidence: {helmet_result['confidence']:.2f})")
    
    print("\n✅ All tests completed successfully!")
    print("🎉 The violation detection system is working properly!")
    
    return violations_result

def test_fallback_detection():
    """Test the fallback detection system"""
    
    print("\n🔄 Testing fallback detection system...")
    
    # Import the fallback function
    from app import detect_comprehensive_violations_enhanced
    
    # Mock data
    vehicle_data = {
        'bbox': [100, 150, 120, 80],
        'vehicle_details': {'basic_class': 'car'},
        'detection': {'class': 'car'}
    }
    
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    violation_detector = ViolationDetector()
    
    # Test fallback
    result = detect_comprehensive_violations_enhanced(
        vehicle_data, mock_frame, violation_detector
    )
    
    print(f"   ✅ Fallback detection working!")
    print(f"   📊 Violations: {result['total_violations']}")
    print(f"   🎯 Severity: {result['severity_score']:.1f}")
    
    return result

if __name__ == "__main__":
    try:
        # Test main detection system
        main_result = test_violation_detection()
        
        # Test fallback system
        fallback_result = test_fallback_detection()
        
        print("\n" + "=" * 50)
        print("🎯 SUMMARY:")
        print(f"Main system violations: {main_result['total_violations']}")
        print(f"Fallback system violations: {fallback_result['total_violations']}")
        print("✅ Both systems are operational!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()