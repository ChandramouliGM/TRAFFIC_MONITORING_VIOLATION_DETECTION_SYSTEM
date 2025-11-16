#!/usr/bin/env python3
"""
Simple email test without Unicode characters
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime

def load_env_file():
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("ERROR: .env file not found")
        return {}
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def test_email():
    """Test email sending functionality"""
    
    print("SRM Traffic Violation Detection System - Email Test")
    print("=" * 50)
    
    # Load environment variables
    env_vars = load_env_file()
    
    # Get email configuration
    email_address = os.environ.get('EMAIL_ID')
    email_password = os.environ.get('EMAIL_APP_PASSWORD')
    recipient_email = os.environ.get('ALERT_RECIPIENT_EMAIL') or email_address
    smtp_server = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('EMAIL_SMTP_PORT', '465'))
    
    print(f"Email Configuration:")
    print(f"   From: {email_address}")
    print(f"   To: {recipient_email}")
    print(f"   SMTP: {smtp_server}:{smtp_port}")
    
    if not email_address or not email_password:
        print("\nERROR: Email configuration missing!")
        print("Required in .env file:")
        print("   EMAIL_ID=your_email@gmail.com")
        print("   EMAIL_APP_PASSWORD=your_app_password")
        return False
    
    # Create email message
    message = EmailMessage()
    message['Subject'] = 'SRM Traffic System - Test Alert'
    message['From'] = email_address
    message['To'] = recipient_email
    
    # Email content
    email_content = f"""
SRM Traffic Violation Detection System - Test Alert

This is a test email sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you receive this email, the alert system is working correctly!

System Details:
- Email Provider: Gmail SMTP
- Configuration: .env file
- Test Status: SUCCESS

SRM Institute of Science & Technology
Final Year Project - Traffic Violation Detection System

Best regards,
SRM Traffic System
    """
    
    message.set_content(email_content)
    
    # Send email
    print("\nSending test email...")
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(email_address, email_password)
            server.send_message(message)
        
        print("SUCCESS: Test email sent successfully!")
        print(f"Check your inbox: {recipient_email}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to send email: {str(e)}")
        return False

def test_violation_email():
    """Test violation alert email"""
    
    print("\nSending Sample Violation Alert...")
    
    # Load environment variables
    load_env_file()
    
    email_address = os.environ.get('EMAIL_ID')
    email_password = os.environ.get('EMAIL_APP_PASSWORD')
    recipient_email = os.environ.get('ALERT_RECIPIENT_EMAIL') or email_address
    smtp_server = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('EMAIL_SMTP_PORT', '465'))
    
    if not email_address or not email_password:
        print("ERROR: Email not configured")
        return False
    
    # Create violation alert email
    message = EmailMessage()
    message['Subject'] = 'TRAFFIC VIOLATION ALERT - Speeding Detected'
    message['From'] = email_address
    message['To'] = recipient_email
    
    violation_content = f"""
TRAFFIC VIOLATION DETECTED

Type: Speeding
License Plate: KA01AB1234
Speed: 85 km/h (Limit: 50 km/h)
Confidence: 95%
Location: Main Street Junction
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMMEDIATE ACTION REQUIRED

This violation was detected by the SRM Traffic Violation Detection System.
Please review and take appropriate enforcement action.

SRM Institute of Science & Technology
Final Year Project - Traffic Violation Detection System
    """
    
    message.set_content(violation_content)
    
    # Send violation email
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(email_address, email_password)
            server.send_message(message)
        
        print("SUCCESS: Violation alert email sent successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to send violation email: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Test basic email
        email_success = test_email()
        
        if email_success:
            # Test violation alert
            test_violation_email()
            
            print("\n" + "=" * 50)
            print("SUCCESS: All email tests completed successfully!")
            print("Check your email inbox for test messages")
            print("\nThe email alert system is working correctly!")
        else:
            print("\nERROR: Email configuration needs to be fixed")
            print("Please check your .env file settings")
            
    except Exception as e:
        print(f"\nERROR: Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()