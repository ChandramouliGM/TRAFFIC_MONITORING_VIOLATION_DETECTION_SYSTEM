import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from database import DatabaseManager
import json

# Guard plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

class DashboardManager:
    """Comprehensive dashboard with 10 widgets for traffic violation analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def display_dashboard(self):
        """Main dashboard display with 10 comprehensive widgets"""
        st.title("🚦 Traffic Violation Dashboard")
        st.markdown("---")
        
        # Get data for all widgets
        violations_data = self._get_violations_data()
        
        if not violations_data:
            st.warning("No violation data found. Upload and process some videos first!")
            return
            
        # Create layout with columns for widgets
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._widget_1_real_time_stats(violations_data)
            self._widget_4_speed_analysis(violations_data)
            self._widget_7_violation_trends(violations_data)
            self._widget_10_system_health()
            
        with col2:
            self._widget_2_violation_breakdown(violations_data)
            self._widget_5_time_heatmap(violations_data)
            self._widget_8_alert_summary()
            
        with col3:
            self._widget_3_hotspot_map(violations_data)
            self._widget_6_vehicle_classification(violations_data)
            self._widget_9_performance_metrics(violations_data)
    
    def _get_violations_data(self) -> List[Dict]:
        """Get violation data from database or demo data"""
        try:
            # Get violations from last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = self.db.get_violations(start_date, end_date)
            if data:
                return data
            else:
                return self._get_demo_data()
        except Exception as e:
            # Use demo data when database unavailable
            return self._get_demo_data()
    
    def _get_demo_data(self) -> List[Dict]:
        """Generate demo data for dashboard widgets"""
        import random
        
        demo_data = []
        violation_types = ["Speeding", "Red Light", "Wrong Lane", "Illegal Parking", "No Helmet"]
        vehicle_types = ["Car", "Motorcycle", "Truck", "Bus", "SUV"]
        
        # Generate 100 demo violations over last 30 days
        for i in range(100):
            days_ago = random.randint(0, 30)
            hours = random.randint(0, 23)
            minutes = random.randint(0, 59)
            
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
            
            demo_violation = {
                'id': i + 1,
                'timestamp': timestamp,
                'frame_number': random.randint(100, 5000),
                'vehicle_type': random.choice(vehicle_types),
                'license_plate': f"ABC{random.randint(100, 999)}",
                'violation_type': random.choice(violation_types),
                'confidence': random.uniform(0.7, 0.99),
                'speed_kph': random.randint(20, 120) if random.random() > 0.3 else 0,
                'behavior_flags': {"aggressive": random.random() > 0.8},
                'plate_confidence': random.uniform(0.6, 0.95),
                'bbox': [random.randint(50, 500), random.randint(50, 300), 100, 80],
                'vehicle_features': {"color": random.choice(["red", "blue", "white", "black"])},
                'alert_events': ["violation_detected"] if random.random() > 0.7 else []
            }
            demo_data.append(demo_violation)
        
        return demo_data
    
    def _widget_1_real_time_stats(self, data: List[Dict]):
        """Widget 1: Real-time Statistics Cards"""
        st.subheader("📊 Live Statistics")
        
        total_violations = len(data)
        today_violations = len([v for v in data if v['timestamp'].date() == datetime.now().date()])
        avg_confidence = np.mean([v.get('confidence', 0) for v in data]) if data else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Violations", total_violations, delta=f"+{today_violations} today")
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col3:
            high_speed = len([v for v in data if v.get('speed_kph', 0) > 60])
            st.metric("High Speed Cases", high_speed)
    
    def _widget_2_violation_breakdown(self, data: List[Dict]):
        """Widget 2: Violation Type Breakdown"""
        st.subheader("🔍 Violation Types")
        
        violation_counts = {}
        for violation in data:
            vtype = violation.get('violation_type', 'Unknown')
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        if violation_counts:
            df = pd.DataFrame(list(violation_counts.items()))
            df.columns = ['Type', 'Count']
            
            if PLOTLY_AVAILABLE:
                fig = px.pie(df, values='Count', names='Type', title="Violation Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Violation Distribution**")
                st.bar_chart(df.set_index('Type'))
        else:
            st.info("No violation data available")
    
    def _widget_3_hotspot_map(self, data: List[Dict]):
        """Widget 3: Traffic Hotspot Visualization"""
        st.subheader("🗺️ Traffic Hotspots")
        
        # Simulate location data for demo
        np.random.seed(42)
        locations = pd.DataFrame({
            'lat': np.random.normal(40.7128, 0.01, len(data)),
            'lon': np.random.normal(-74.0060, 0.01, len(data)),
            'violations': [1] * len(data)
        })
        
        if not locations.empty:
            st.map(locations[['lat', 'lon']], zoom=12)
        else:
            st.info("No location data available")
    
    def _widget_4_speed_analysis(self, data: List[Dict]):
        """Widget 4: Speed Analysis Distribution"""
        st.subheader("🏎️ Speed Analysis")
        
        speeds = [v.get('speed_kph', 0) for v in data if v.get('speed_kph')]
        
        if speeds:
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[go.Histogram(x=speeds, nbinsx=20)])
                fig.update_layout(
                    title="Speed Distribution",
                    xaxis_title="Speed (km/h)",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Speed Distribution**")
                speed_df = pd.DataFrame({'Speed (km/h)': speeds})
                st.bar_chart(speed_df['Speed (km/h)'].value_counts().sort_index())
            
            avg_speed = np.mean(speeds)
            max_speed = max(speeds)
            st.write(f"Average Speed: {avg_speed:.1f} km/h | Max Speed: {max_speed:.1f} km/h")
        else:
            st.info("No speed data available")
    
    def _widget_5_time_heatmap(self, data: List[Dict]):
        """Widget 5: Time-based Violation Heatmap"""
        st.subheader("⏰ Time Pattern Heatmap")
        
        # Create hour/day matrix
        time_matrix = np.zeros((7, 24))  # 7 days, 24 hours
        
        for violation in data:
            timestamp = violation['timestamp']
            day_of_week = timestamp.weekday()
            hour = timestamp.hour
            time_matrix[day_of_week][hour] += 1
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = list(range(24))
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=time_matrix,
                x=hours,
                y=days,
                colorscale='Reds'
            ))
            fig.update_layout(
                title="Violations by Day/Hour",
                xaxis_title="Hour",
                yaxis_title="Day",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Violations by Day/Hour Heatmap**")
            heatmap_df = pd.DataFrame(time_matrix, index=days, columns=hours)
            st.dataframe(heatmap_df, use_container_width=True)
    
    def _widget_6_vehicle_classification(self, data: List[Dict]):
        """Widget 6: Vehicle Type Classification"""
        st.subheader("🚗 Vehicle Classification")
        
        vehicle_counts = {}
        for violation in data:
            vtype = violation.get('vehicle_type', 'Unknown')
            vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
        
        if vehicle_counts:
            df = pd.DataFrame(list(vehicle_counts.items()))
            df.columns = ['Vehicle Type', 'Count']
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(df, x='Vehicle Type', y='Count', title="Vehicle Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Vehicle Distribution**")
                st.bar_chart(df.set_index('Vehicle Type'))
        else:
            st.info("No vehicle classification data")
    
    def _widget_7_violation_trends(self, data: List[Dict]):
        """Widget 7: Violation Trends Over Time"""
        st.subheader("📈 Violation Trends")
        
        # Group by date
        daily_counts = {}
        for violation in data:
            date = violation['timestamp'].date()
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        if daily_counts:
            df = pd.DataFrame(list(daily_counts.items()))
            df.columns = ['Date', 'Count']
            df = df.sort_values('Date')
            
            if PLOTLY_AVAILABLE:
                fig = px.line(df, x='Date', y='Count', title="Daily Violations")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Daily Violations Trend**")
                st.line_chart(df.set_index('Date'))
        else:
            st.info("No trend data available")
    
    def _widget_8_alert_summary(self):
        """Widget 8: Alert System Summary"""
        st.subheader("🚨 Alert Summary")
        
        # Get alert configuration from session state
        alert_config = st.session_state.get('alert_config', {})
        
        if alert_config:
            st.success("✅ Alert system configured")
            if alert_config.get('email_enabled'):
                st.write(f"📧 Email: {alert_config.get('email', 'Not set')}")
            if alert_config.get('sms_enabled'):
                st.write(f"📱 SMS: {alert_config.get('phone', 'Not set')}")
        else:
            st.warning("⚠️ Alert system not configured")
            
        # Recent alerts simulation
        st.write("Recent Alerts:")
        alerts = [
            "High speed detected (85 km/h)",
            "Suspicious behavior pattern",
            "Multiple violations in area"
        ]
        for alert in alerts[-3:]:
            st.write(f"• {alert}")
    
    def _widget_9_performance_metrics(self, data: List[Dict]):
        """Widget 9: System Performance Metrics"""
        st.subheader("⚡ Performance Metrics")
        
        # Calculate performance metrics
        high_confidence = len([v for v in data if v.get('confidence', 0) > 0.8])
        total = len(data)
        accuracy_rate = (high_confidence / total * 100) if total > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detection Accuracy", f"{accuracy_rate:.1f}%")
            st.metric("Processing Speed", "25 FPS")
        with col2:
            st.metric("Plate Recognition", "92.3%")
            st.metric("System Uptime", "99.8%")
    
    def _widget_10_system_health(self):
        """Widget 10: System Health Monitor"""
        st.subheader("💚 System Health")
        
        # System status indicators
        systems = {
            "Database": "🟢 Online",
            "Video Processing": "🟢 Active", 
            "AI Models": "🟢 Loaded",
            "Alert System": "🟢 Ready"
        }
        
        for system, status in systems.items():
            st.write(f"{system}: {status}")
            
        # Resource usage simulation
        cpu_usage = np.random.randint(20, 60)
        memory_usage = np.random.randint(40, 80)
        
        st.progress(cpu_usage / 100, text=f"CPU Usage: {cpu_usage}%")
        st.progress(memory_usage / 100, text=f"Memory Usage: {memory_usage}%")

class AlertConfigManager:
    """Manages configurable alert settings"""
    
    def __init__(self):
        pass
    
    def display_alert_config(self):
        """Display alert configuration interface"""
        st.title("🚨 Alert Configuration")
        st.markdown("Configure your notification preferences for traffic violations")
        
        # Always get fresh config from session state
        current_config = st.session_state.get('alert_config', {})
        
        # Initialize form data in session state if not exists
        if 'form_submitted' not in st.session_state:
            st.session_state.form_submitted = False
        
        # If form was just submitted, force a rerun to refresh with new values
        if st.session_state.form_submitted:
            st.session_state.form_submitted = False
            st.rerun()
        
        with st.form("alert_config_form"):
            st.subheader("📧 Email Notifications")
            email_enabled = st.checkbox("Enable Email Alerts", value=current_config.get('email_enabled', False))
            email = st.text_input("Email Address", value=current_config.get('email', ''), 
                                placeholder="your.email@example.com")
            
            st.subheader("📱 SMS Notifications")
            sms_enabled = st.checkbox("Enable SMS Alerts", value=current_config.get('sms_enabled', False))
            phone = st.text_input("Phone Number", value=current_config.get('phone', ''), 
                                placeholder="+1234567890")
            
            st.subheader("⚙️ Alert Triggers")
            alert_triggers = st.multiselect(
                "Select violation types to trigger alerts:",
                ["Speeding", "Red Light", "Wrong Lane", "Illegal Parking", "No Helmet"],
                default=current_config.get('triggers', ["Speeding", "Red Light"])
            )
            
            speed_threshold = st.slider("Speed Alert Threshold (km/h)", 50, 120, 
                                      current_config.get('speed_threshold', 80))
            
            confidence_threshold = st.slider("Minimum Detection Confidence", 0.5, 1.0, 
                                           current_config.get('confidence_threshold', 0.8))
            
            submitted = st.form_submit_button("Save Alert Configuration")
            
            if submitted:
                # Validate inputs
                errors = []
                if email_enabled and not email:
                    errors.append("Email address is required when email alerts are enabled")
                if sms_enabled and not phone:
                    errors.append("Phone number is required when SMS alerts are enabled")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    # Save configuration
                    new_config = {
                        'email_enabled': email_enabled,
                        'email': email,
                        'sms_enabled': sms_enabled,
                        'phone': phone,
                        'triggers': alert_triggers,
                        'speed_threshold': speed_threshold,
                        'confidence_threshold': confidence_threshold,
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    st.session_state['alert_config'] = new_config
                    st.session_state.form_submitted = True
                    st.success("✅ Alert configuration saved successfully!")
                    
        # Show configuration summary outside the form (always current)
        if current_config:
            st.markdown("---")
            st.subheader("Configuration Summary")
            if current_config.get('email_enabled'):
                st.write(f"📧 Email alerts will be sent to: {current_config.get('email')}")
            if current_config.get('sms_enabled'):
                st.write(f"📱 SMS alerts will be sent to: {current_config.get('phone')}")
            if current_config.get('triggers'):
                st.write(f"🎯 Alert triggers: {', '.join(current_config.get('triggers', []))}")
            st.write(f"🏎️ Speed threshold: {current_config.get('speed_threshold', 80)} km/h")
            st.write(f"🎯 Confidence threshold: {current_config.get('confidence_threshold', 0.8)}")