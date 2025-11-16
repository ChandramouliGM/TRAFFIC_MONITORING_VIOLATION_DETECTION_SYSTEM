#!/usr/bin/env python3
"""
Quick email test script for SRM Traffic Violation Detection System
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alert_system import AlertNotificationSystem

def test_email_sending():
    """Test email sending functionality"""
    
    print("🚦 Testing Email Alert System")
    print("=" * 40)
    
    # Initialize alert system
    alert_system = AlertNotificationSystem()
    
    # Check configuration
    print("📧 Email Configuration:")
    print(f"   SMTP Configured: {alert_system.smtp_configured}")
    print(f"   Email Address: {alert_system.email_address}")
    print(f"   Recipient: {alert_system.default_recipient_email}")
    print(f"   SMTP Server: {alert_system.smtp_server}:{alert_system.smtp_port}")
    
    if not alert_system.smtp_configured:
        print("\n❌ Email not configured!")
        print("Check your .env file for EMAIL_ID and EMAIL_APP_PASSWORD")
        return False
    
    # Send test email
    print("\n📤 Sending test email...")
    
    test_message = f"""
🚦 SRM Traffic Violation Detection System - Test Alert

This is a test email sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you receive this email, the alert system is working correctly!

System Details:
- Email Provider: Gmail SMTP
- Configuration: .env file
- Test Status: SUCCESS

Best regards,
SRM Traffic Violation Detection System
    """
    
    success = alert_system.send_email_alert(
        to_email=None,  # Uses default recipient
        subject="🚦 SRM Traffic System - Test Alert",
        message=test_message
    )
    
    if success:
        print("✅ Test email sent successfully!")
        print(f"📧 Check your inbox: {alert_system.default_recipient_email}")
        return True
    else:
        print("❌ Failed to send test email")
        return False

def test_violation_alert():
    """Test violation alert email"""
    
    print("\n🚨 Testing Violation Alert Email...")
    
    alert_system = AlertNotificationSystem()
    
    # Sample violation data
    violation_data = {
        'violation_type': 'Speeding',
        'license_plate': 'KA01AB1234',
        'confidence': 0.95,
        'speed_kph': 85,
        'timestamp': datetime.now(),
        'location': 'Main Street Junction'
    }
    
    # Alert configuration
    alert_config = {
        'email_enabled': True,
        'triggers': ['Speeding'],
        'confidence_threshold': 0.8,
        'speed_threshold': 70
    }
    
    # Process violation alert
    alert_system.process_violation_alert(
        violation_data=violation_data,
        alert_config=alert_config
    )
    
    print("✅ Violation alert processed!")

if __name__ == "__main__":
    try:
        # Test basic email
        email_success = test_email_sending()
        
        if email_success:
            # Test violation alert
            test_violation_alert()
            
            print("\n" + "=" * 40)
            print("🎉 All email tests completed!")
            print("📧 Check your email inbox for test messages")
        else:
            print("\n❌ Email configuration needs to be fixed")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()