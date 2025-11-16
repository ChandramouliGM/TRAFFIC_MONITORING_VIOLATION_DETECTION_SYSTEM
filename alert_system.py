# Alert system using Twilio and SendGrid integrations
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import streamlit as st
from datetime import datetime

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    DOTENV_AVAILABLE = False

if DOTENV_AVAILABLE:
    load_dotenv()

# Twilio SMS integration
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    Client = None
    TWILIO_AVAILABLE = False

# SendGrid email integration  
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SendGridAPIClient = None
    Mail = None
    Email = None
    To = None
    Content = None
    SENDGRID_AVAILABLE = False

class AlertNotificationSystem:
    """Configurable alert system for traffic violations using SMTP email (Twilio/SendGrid optional)."""
    
    def __init__(self):
        self.smtp_only = os.environ.get("EMAIL_SMTP_ONLY", "true").lower() in ("1", "true", "yes")
        self.sms_alerts_enabled = os.environ.get("SMS_ALERTS_ENABLED", "false").lower() in ("1", "true", "yes")

        self.twilio_available = TWILIO_AVAILABLE and self.sms_alerts_enabled
        self.sendgrid_available = SENDGRID_AVAILABLE and (not self.smtp_only)
        
        # Initialize Twilio client if available
        if self.twilio_available:
            self.twilio_account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN") 
            self.twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
            
            if all([self.twilio_account_sid, self.twilio_auth_token, self.twilio_phone_number]):
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                self.twilio_configured = True
            else:
                self.twilio_configured = False
        else:
            self.twilio_configured = False
            
        # Initialize SendGrid client if available and not SMTP-only
        if self.sendgrid_available:
            self.sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")
            if self.sendgrid_api_key:
                self.sendgrid_client = SendGridAPIClient(self.sendgrid_api_key)
                self.sendgrid_configured = True
            else:
                self.sendgrid_configured = False
        else:
            self.sendgrid_configured = False

        # SMTP/Gmail configuration from .env (app password)
        self.email_address = (
            os.environ.get("EMAIL_ID")
            or os.environ.get("EMAIL_ADDRESS")
            or os.environ.get("SMTP_EMAIL")
        )
        self.email_app_password = (
            os.environ.get("EMAIL_APP_PASSWORD")
            or os.environ.get("EMAIL_PASSWORD")
            or os.environ.get("SMTP_EMAIL_PASSWORD")
        )
        self.smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        try:
            self.smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
        except ValueError:
            self.smtp_port = 465
        self.smtp_use_tls = os.environ.get("EMAIL_SMTP_USE_TLS", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self.smtp_configured = bool(self.email_address and self.email_app_password)
        self.default_recipient_email = (
            os.environ.get("ALERT_RECIPIENT_EMAIL")
            or os.environ.get("ALERT_TO_EMAIL")
            or self.email_address
        )
    
    def send_sms_alert(self, phone_number: str, message: str) -> bool:
        """Send SMS alert using Twilio if explicitly enabled."""
        if not self.sms_alerts_enabled:
            st.info("SMS alerts are disabled. Set SMS_ALERTS_ENABLED=true in .env to enable Twilio delivery.")
            return False

        if not self.twilio_configured:
            st.warning("SMS alerts not configured. Please set up Twilio credentials or keep SMS alerts disabled.")
            return False
            
        try:
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone_number,
                to=phone_number
            )
            st.success(f"SMS alert sent successfully! Message SID: {message_obj.sid}")
            return True
        except Exception as e:
            st.error(f"Failed to send SMS: {str(e)}")
            return False
    
    def send_email_alert(
        self,
        to_email: Optional[str] = None,
        subject: str = "Traffic Violation Alert",
        message: str = "",
        from_email: Optional[str] = None,
    ) -> bool:
        """Send email alert using SMTP credentials from .env (preferred) or SendGrid fallback."""
        recipient = to_email or self.default_recipient_email
        if not recipient:
            st.warning("No recipient email configured. Set ALERT_RECIPIENT_EMAIL in .env or enable email alerts in settings.")
            return False

        html_body = self._format_email_html(message)

        if self.smtp_configured:
            email_message = self._build_email_message(
                to_email=recipient,
                subject=subject,
                text_body=message or "Traffic violation detected.",
                html_body=html_body,
                from_email=from_email or self.email_address,
            )
            return self._send_email_via_smtp(email_message)

        if self.sendgrid_configured:
            sender = from_email or self.email_address or "alerts@trafficguard.com"
            try:
                mail = Mail(
                    from_email=Email(sender),
                    to_emails=To(recipient),
                    subject=subject,
                    html_content=Content("text/html", html_body),
                )

                response = self.sendgrid_client.send(mail)
                st.success(f"Email alert sent successfully via SendGrid! Status: {response.status_code}")
                return True
            except Exception as exc:
                st.error(f"Failed to send email via SendGrid: {str(exc)}")
                return False

        st.warning("Email alerts not configured. Provide EMAIL_ID and EMAIL_APP_PASSWORD in your .env.")
        return False
    
    def _format_email_html(self, message: str) -> str:
        """Format message as HTML email"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #ff4444; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f0f0f0; padding: 10px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚦 TrafficGuard Alert</h2>
            </div>
            <div class="content">
                <p>{message}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="footer">
                <p>This is an automated alert from TrafficGuard Traffic Violation Detection System</p>
            </div>
        </body>
        </html>
        """

    def _build_email_message(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None,
        from_email: Optional[str] = None,
    ) -> EmailMessage:
        """Construct an EmailMessage with both text and HTML content."""
        sender = from_email or self.email_address or "alerts@trafficguard.com"

        email_message = EmailMessage()
        email_message["Subject"] = subject
        email_message["From"] = sender
        email_message["To"] = to_email
        email_message.set_content(text_body)

        if html_body:
            email_message.add_alternative(html_body, subtype="html")

        return email_message

    def _send_email_via_smtp(self, email_message: EmailMessage) -> bool:
        """Send email using SMTP credentials loaded from .env."""
        if not self.smtp_configured:
            st.warning("SMTP email not configured. Check EMAIL_ID and EMAIL_APP_PASSWORD in your .env file.")
            return False

        try:
            # Create SSL context with certificate verification disabled for compatibility
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            if self.smtp_use_tls:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.email_address, self.email_app_password)
                    server.send_message(email_message)
            else:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.email_address, self.email_app_password)
                    server.send_message(email_message)

            st.success("Email alert sent successfully via SMTP!")
            return True
        except Exception as exc:
            st.error(f"Failed to send email via SMTP: {str(exc)}")
            return False

    def _prepare_attachments(
        self, attachment_paths: Optional[Sequence[Union[str, Path]]]
    ) -> List[Tuple[bytes, str, str, str]]:
        """Read attachment files and prepare them for email sending."""
        prepared: List[Tuple[bytes, str, str, str]] = []
        if not attachment_paths:
            return prepared

        for attachment in attachment_paths:
            path_obj = Path(attachment)
            if not path_obj.exists():
                st.warning(f"Attachment not found: {path_obj}")
                continue

            try:
                with path_obj.open("rb") as file_obj:
                    data = file_obj.read()
            except Exception as exc:
                st.error(f"Could not read attachment {path_obj}: {str(exc)}")
                continue

            mime_type, _ = mimetypes.guess_type(str(path_obj))
            if mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"

            prepared.append((data, maintype, subtype, path_obj.name))

        return prepared

    def send_violation_email_with_attachment(
        self,
        to_email: Optional[str] = None,
        subject: str = "Traffic Violation Evidence",
        message: str = "",
        attachment_paths: Optional[Sequence[Union[str, Path]]] = None,
        from_email: Optional[str] = None,
    ) -> bool:
        """Send a violation email with attachments such as snapshots or video clips."""
        recipient = to_email or self.default_recipient_email
        if not recipient:
            st.warning("Attachment emails require ALERT_RECIPIENT_EMAIL or an explicit recipient.")
            return False

        if not self.smtp_configured:
            st.warning("Attachment emails require EMAIL_ID and EMAIL_APP_PASSWORD in your .env file.")
            return False

        attachments = self._prepare_attachments(attachment_paths)
        html_body = self._format_email_html(message or "Please review the attached violation evidence.")
        email_message = self._build_email_message(
            to_email=recipient,
            subject=subject,
            text_body=message or "Traffic violation detected. See attachments.",
            html_body=html_body,
            from_email=from_email or self.email_address,
        )

        for data, maintype, subtype, filename in attachments:
            email_message.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

        return self._send_email_via_smtp(email_message)
    
    def process_violation_alert(
        self,
        violation_data: Dict,
        alert_config: Optional[Dict],
        attachment_paths: Optional[Sequence[Union[str, Path]]] = None,
    ) -> None:
        """Process a violation and send alerts if configured"""
        alert_config = alert_config or {}
        if not alert_config and not self.default_recipient_email:
            return
            
        # Check if violation meets alert criteria
        if not self._should_trigger_alert(violation_data, alert_config):
            return
            
        # Generate alert message
        message = self._generate_alert_message(violation_data)
        subject = f"Traffic Violation Alert: {violation_data.get('violation_type', 'Unknown')}"
        
        # Send SMS if enabled
        if alert_config.get('sms_enabled') and alert_config.get('phone'):
            self.send_sms_alert(alert_config['phone'], message)
            
        # Prepare attachments from violation data when available
        attachments_to_send: Optional[Sequence[Union[str, Path]]] = attachment_paths
        if attachments_to_send is None:
            for key in ("attachments", "attachment_paths", "evidence_files", "evidence_paths"):
                candidate = violation_data.get(key)
                if candidate:
                    if isinstance(candidate, (list, tuple, set)):
                        attachments_to_send = list(candidate)
                    else:
                        attachments_to_send = [candidate]
                    break

        # Determine email routing
        email_enabled = alert_config.get('email_enabled')
        if email_enabled is None:
            email_enabled = bool(self.default_recipient_email)

        recipient_email = alert_config.get('email') or self.default_recipient_email

        if email_enabled and recipient_email:
            if attachments_to_send:
                self.send_violation_email_with_attachment(
                    recipient_email,
                    subject,
                    message,
                    attachment_paths=attachments_to_send,
                )
            else:
                self.send_email_alert(recipient_email, subject, message)
        elif email_enabled and not recipient_email:
            st.warning("Email alerts enabled but no recipient email is configured.")
    
    def _should_trigger_alert(self, violation_data: Dict, alert_config: Dict) -> bool:
        """Determine if violation should trigger an alert"""
        # Check violation type
        violation_type = violation_data.get('violation_type', '')
        if violation_type not in alert_config.get('triggers', []):
            return False
            
        # Check confidence threshold
        confidence = violation_data.get('confidence', 0)
        if confidence < alert_config.get('confidence_threshold', 0.8):
            return False
            
        # Check speed threshold for speeding violations
        if violation_type == 'Speeding':
            speed = violation_data.get('speed_kph', 0)
            if speed < alert_config.get('speed_threshold', 80):
                return False
                
        return True
    
    def _generate_alert_message(self, violation_data: Dict) -> str:
        """Generate alert message for violation"""
        violation_type = violation_data.get('violation_type', 'Unknown')
        license_plate = violation_data.get('license_plate', 'Unknown')
        confidence = violation_data.get('confidence', 0)
        speed = violation_data.get('speed_kph', 0)
        timestamp = violation_data.get('timestamp', datetime.now())
        
        message = f"🚨 TRAFFIC VIOLATION DETECTED\n\n"
        message += f"Type: {violation_type}\n"
        message += f"License Plate: {license_plate}\n"
        message += f"Confidence: {confidence:.1%}\n"
        
        if speed > 0:
            message += f"Speed: {speed:.1f} km/h\n"
            
        message += f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"\nTake appropriate action immediately."
        
        return message
    

    def test_alert_system(self, alert_config: Dict) -> None:
        """Test the alert system configuration."""
        st.subheader("Test Alert System")

        alert_config = alert_config or {}
        test_violation = {
            'violation_type': 'Traffic Violation Alert',
            'license_plate': 'KA-01-HR-6789',
            'confidence': 0.95,
            'speed_kph': 85,
            'timestamp': datetime.now()
        }

        col1, col2 = st.columns(2)

        with col1:
            sms_ready = (
                self.sms_alerts_enabled
                and alert_config.get('sms_enabled')
                and alert_config.get('phone')
            )
            if st.button("Test SMS Alert", disabled=not sms_ready):
                message = (
                    "Test SMS from TrafficGuard system. If you receive this, SMS alerts "
                    "are working correctly!"
                )
                success = self.send_sms_alert(alert_config['phone'], message)
                if success:
                    st.success("SMS test completed!")
            if not sms_ready:
                if not self.sms_alerts_enabled:
                    st.info("SMS alerts disabled via .env (SMS_ALERTS_ENABLED=false).")
                else:
                    st.warning("SMS not configured or enabled")

        with col2:
            default_email_target = alert_config.get('email') or self.default_recipient_email
            email_enabled = alert_config.get('email_enabled')
            if email_enabled is None:
                email_enabled = bool(default_email_target)
            email_ready = email_enabled and bool(default_email_target) and self.smtp_configured

            if st.button("Test Email Alert", disabled=not email_ready):
                message = (
                    "Test email from TrafficGuard system. If you receive this, email alerts "
                    "are working correctly!"
                )
                success = self.send_email_alert(default_email_target, "TrafficGuard Test Alert", message)
                if success:
                    st.success("Email test completed!")
            if not email_ready:
                if not self.smtp_configured:
                    st.error("SMTP email not configured. Update your .env with EMAIL_ID and EMAIL_APP_PASSWORD.")
                elif not default_email_target:
                    st.warning("Email recipient not configured.")
                elif not email_enabled:
                    st.info("Email alerts disabled in settings.")


    def get_system_status(self) -> Dict:
        """Get alert system status"""
        return {
            'twilio_available': self.twilio_available,
            'twilio_configured': self.twilio_configured,
            'sendgrid_available': self.sendgrid_available,
            'sendgrid_configured': self.sendgrid_configured,
            'smtp_configured': self.smtp_configured,
            'sms_enabled': self.sms_alerts_enabled,
            'smtp_only': self.smtp_only,
            'email_channel': 'smtp' if self.smtp_configured else ('sendgrid' if self.sendgrid_configured else 'none')
        }
    
    def display_system_status(self):
        """Display alert system status"""
        st.subheader("📊 Alert System Status")
        
        status = self.get_system_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SMS (Twilio)**")
            if not status['sms_enabled']:
                st.info("SMS alerts disabled via .env.")
            elif status['twilio_configured']:
                st.success("✅ SMS alerts ready")
            elif status['twilio_available']:
                st.warning("⚠️ Twilio available but not configured")
            else:
                st.error("❌ Twilio not available")
                
        with col2:
            st.write("**Email (.env SMTP / SendGrid)**")
            if status['smtp_configured']:
                st.success("✅ Email alerts ready via SMTP (.env credentials)")
            elif status['sendgrid_configured']:
                st.warning("⚠️ Using SendGrid fallback - configure .env for direct email alerts")
            elif status['sendgrid_available']:
                st.warning("⚠️ Email integrations installed but credentials missing")
            else:
                st.error("❌ Email integrations not available")
                
        email_ready = status['smtp_configured'] or status['sendgrid_configured']
        sms_ready = status['sms_enabled'] and status['twilio_configured']
        if not (email_ready or sms_ready):
            st.info("💡 Configure Gmail SMTP in .env or enable Twilio SMS to activate alerts.")
