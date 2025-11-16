import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
import io
import streamlit as st
from database import DatabaseManager


class ReportsGenerator:
    """Advanced reporting system for traffic violation analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize reports generator"""
        self.db_manager = db_manager
        self.report_types = {
            'daily_summary': 'Daily Violation Summary',
            'trend_analysis': 'Weekly/Monthly Trend Analysis', 
            'vehicle_breakdown': 'Vehicle Type Violation Breakdown',
            'hotspot_analysis': 'Location-wise Violation Hotspot Analysis',
            'time_pattern': 'Time-based Violation Pattern Analysis',
            'violation_history': 'License Plate Violation History',
            'comprehensive': 'Comprehensive Analysis Report'
        }
    
    def generate_daily_summary_report(self, target_date: Optional[date] = None) -> Dict:
        """Generate daily violation summary report"""
        if target_date is None:
            target_date = datetime.now().date()
        
        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = datetime.combine(target_date, datetime.max.time())
        
        # Get violations for the day
        violations = self.db_manager.get_violations(start_datetime, end_datetime)
        
        if not violations:
            return {
                'title': f'Daily Summary Report - {target_date.strftime("%Y-%m-%d")}',
                'summary': {'total_violations': 0, 'unique_vehicles': 0},
                'data': pd.DataFrame(),
                'charts': {}
            }
        
        df = pd.DataFrame(violations)
        
        # Summary statistics
        summary = {
            'date': target_date.strftime("%Y-%m-%d"),
            'total_violations': len(df),
            'unique_vehicles': df['license_plate'].nunique(),
            'violation_types': df['violation_type'].value_counts().to_dict(),
            'vehicle_types': df['vehicle_type'].value_counts().to_dict(),
            'peak_hour': df['timestamp'].dt.hour.mode().iloc[0] if not df.empty else 0,
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0
        }
        
        # Hourly distribution
        df['hour'] = df['timestamp'].dt.hour
        hourly_dist = df.groupby('hour').size().reset_index(name='violations')
        
        return {
            'title': f'Daily Summary Report - {target_date.strftime("%Y-%m-%d")}',
            'summary': summary,
            'data': df,
            'hourly_distribution': hourly_dist,
            'charts': {
                'violations_by_type': df['violation_type'].value_counts(),
                'violations_by_hour': hourly_dist.set_index('hour')['violations']
            }
        }
    
    def generate_trend_analysis_report(self, days: int = 30) -> Dict:
        """Generate trend analysis report for specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        violations = self.db_manager.get_violations(start_date, end_date)
        
        if not violations:
            return {
                'title': f'{days}-Day Trend Analysis Report',
                'data': pd.DataFrame(),
                'trends': {}
            }
        
        df = pd.DataFrame(violations)
        df['date'] = df['timestamp'].dt.date
        
        # Daily trends
        daily_trends = df.groupby('date').agg({
            'id': 'count',
            'license_plate': 'nunique',
            'violation_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'confidence': 'mean'
        }).rename(columns={'id': 'total_violations', 'license_plate': 'unique_vehicles'})
        
        # Weekly trends
        df['week'] = df['timestamp'].dt.isocalendar().week
        weekly_trends = df.groupby('week')['id'].count().reset_index(name='violations')
        
        # Violation type trends
        type_trends = df.groupby(['date', 'violation_type']).size().unstack(fill_value=0)
        
        # Growth calculations
        recent_week = daily_trends.tail(7)['total_violations'].sum()
        previous_week = daily_trends.iloc[-14:-7]['total_violations'].sum() if len(daily_trends) >= 14 else recent_week
        growth_rate = ((recent_week - previous_week) / previous_week * 100) if previous_week > 0 else 0
        
        return {
            'title': f'{days}-Day Trend Analysis Report',
            'period': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'summary': {
                'total_violations': len(df),
                'daily_average': len(df) / days,
                'growth_rate': growth_rate,
                'peak_day': daily_trends.idxmax()['total_violations'],
                'most_common_violation': df['violation_type'].mode().iloc[0] if not df.empty else 'None'
            },
            'data': df,
            'daily_trends': daily_trends,
            'weekly_trends': weekly_trends,
            'type_trends': type_trends
        }
    
    def generate_vehicle_breakdown_report(self) -> Dict:
        """Generate vehicle type violation breakdown report"""
        # Get all violations
        violations = self.db_manager.get_violations()
        
        if not violations:
            return {
                'title': 'Vehicle Type Violation Breakdown Report',
                'data': pd.DataFrame(),
                'breakdown': {}
            }
        
        df = pd.DataFrame(violations)
        
        # Vehicle type analysis
        vehicle_breakdown = df.groupby('vehicle_type').agg({
            'id': 'count',
            'license_plate': 'nunique',
            'violation_type': lambda x: x.value_counts().to_dict(),
            'confidence': 'mean'
        }).rename(columns={'id': 'total_violations', 'license_plate': 'unique_vehicles'})
        
        # Most common violations by vehicle type
        vehicle_violation_matrix = df.groupby(['vehicle_type', 'violation_type']).size().unstack(fill_value=0)
        
        # Risk analysis by vehicle type
        risk_analysis = {}
        for vehicle_type in df['vehicle_type'].unique():
            vehicle_data = df[df['vehicle_type'] == vehicle_type]
            risk_analysis[vehicle_type] = {
                'violation_rate': len(vehicle_data) / df['license_plate'].nunique() * 100,
                'most_common_violation': vehicle_data['violation_type'].mode().iloc[0] if not vehicle_data.empty else 'None',
                'avg_confidence': vehicle_data['confidence'].mean() if 'confidence' in vehicle_data.columns else 0
            }
        
        return {
            'title': 'Vehicle Type Violation Breakdown Report',
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': df,
            'breakdown': vehicle_breakdown,
            'violation_matrix': vehicle_violation_matrix,
            'risk_analysis': risk_analysis
        }
    
    def generate_hotspot_analysis_report(self) -> Dict:
        """Generate location-wise violation hotspot analysis report"""
        violations = self.db_manager.get_violations()
        
        if not violations:
            return {
                'title': 'Location-wise Violation Hotspot Analysis Report',
                'data': pd.DataFrame(),
                'hotspots': []
            }
        
        df = pd.DataFrame(violations)
        
        # Create location grids based on bbox coordinates (if available)
        # For simplified analysis, we'll use frame_number as proxy for location
        location_analysis = {}
        
        if 'frame_number' in df.columns:
            # Group by frame ranges (every 100 frames as a location segment)
            df['location_segment'] = (df['frame_number'] // 100) * 100
            
            hotspot_data = df.groupby('location_segment').agg({
                'id': 'count',
                'violation_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'license_plate': 'nunique',
                'confidence': 'mean'
            }).rename(columns={'id': 'violation_count', 'license_plate': 'unique_vehicles'})
            
            # Identify hotspots (top 20% of locations by violation count)
            threshold = hotspot_data['violation_count'].quantile(0.8)
            hotspots = hotspot_data[hotspot_data['violation_count'] >= threshold].sort_values('violation_count', ascending=False)
            
            # Risk classification
            hotspots['risk_level'] = hotspots['violation_count'].apply(
                lambda x: 'Critical' if x > hotspot_data['violation_count'].quantile(0.95)
                         else 'High' if x > hotspot_data['violation_count'].quantile(0.8)
                         else 'Medium'
            )
        else:
            hotspot_data = pd.DataFrame()
            hotspots = pd.DataFrame()
        
        return {
            'title': 'Location-wise Violation Hotspot Analysis Report',
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': df,
            'hotspot_data': hotspot_data,
            'critical_hotspots': hotspots,
            'summary': {
                'total_locations_analyzed': len(hotspot_data) if not hotspot_data.empty else 0,
                'critical_hotspots': len(hotspots[hotspots['risk_level'] == 'Critical']) if not hotspots.empty else 0,
                'high_risk_locations': len(hotspots[hotspots['risk_level'] == 'High']) if not hotspots.empty else 0
            }
        }
    
    def generate_time_pattern_report(self) -> Dict:
        """Generate time-based violation pattern analysis report"""
        violations = self.db_manager.get_violations()
        
        if not violations:
            return {
                'title': 'Time-based Violation Pattern Analysis Report',
                'data': pd.DataFrame(),
                'patterns': {}
            }
        
        df = pd.DataFrame(violations)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
        
        # Time-based analysis
        patterns = {
            'hourly_distribution': df.groupby('hour').size().to_dict(),
            'daily_distribution': df.groupby('day_of_week').size().to_dict(),
            'monthly_distribution': df.groupby('month').size().to_dict(),
            'weekend_vs_weekday': {
                'weekend': len(df[df['is_weekend']]),
                'weekday': len(df[~df['is_weekend']])
            },
            'peak_analysis': {
                'peak_hour': df['hour'].mode().iloc[0] if not df.empty else 0,
                'peak_day': df['day_of_week'].mode().iloc[0] if not df.empty else 'Unknown',
                'rush_hour_violations': len(df[df['hour'].isin([7, 8, 9, 17, 18, 19])]),
                'night_violations': len(df[df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])])
            }
        }
        
        # Violation type by time analysis
        hourly_violation_types = df.groupby(['hour', 'violation_type']).size().unstack(fill_value=0)
        daily_violation_types = df.groupby(['day_of_week', 'violation_type']).size().unstack(fill_value=0)
        
        return {
            'title': 'Time-based Violation Pattern Analysis Report',
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': df,
            'patterns': patterns,
            'hourly_violation_types': hourly_violation_types,
            'daily_violation_types': daily_violation_types
        }
    
    def generate_violation_history_report(self, license_plate: Optional[str] = None) -> Dict:
        """Generate license plate violation history report"""
        violations = self.db_manager.get_violations()
        
        if not violations:
            return {
                'title': 'License Plate Violation History Report',
                'data': pd.DataFrame(),
                'history': {}
            }
        
        df = pd.DataFrame(violations)
        df = df.dropna(subset=['license_plate'])  # Remove entries without license plates
        
        if license_plate:
            df = df[df['license_plate'] == license_plate]
            title = f'Violation History Report - {license_plate}'
        else:
            title = 'Complete License Plate Violation History Report'
        
        # Vehicle history analysis
        vehicle_history = df.groupby('license_plate').agg({
            'id': 'count',
            'violation_type': lambda x: x.value_counts().to_dict(),
            'timestamp': ['min', 'max'],
            'vehicle_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'confidence': 'mean'
        })
        
        vehicle_history.columns = ['total_violations', 'violation_breakdown', 
                                 'first_violation', 'last_violation', 'vehicle_type', 'avg_confidence']
        
        # Repeat offenders (vehicles with multiple violations)
        repeat_offenders = vehicle_history[vehicle_history['total_violations'] > 1].sort_values('total_violations', ascending=False)
        
        # Risk categorization
        vehicle_history['risk_category'] = vehicle_history['total_violations'].apply(
            lambda x: 'High Risk' if x >= 5 
                     else 'Medium Risk' if x >= 3
                     else 'Low Risk'
        )
        
        return {
            'title': title,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': df,
            'vehicle_history': vehicle_history,
            'repeat_offenders': repeat_offenders,
            'summary': {
                'total_unique_vehicles': len(vehicle_history),
                'repeat_offenders_count': len(repeat_offenders),
                'high_risk_vehicles': len(vehicle_history[vehicle_history['risk_category'] == 'High Risk'])
            }
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report combining all aspects"""
        # Get base data
        violations = self.db_manager.get_violations()
        
        if not violations:
            return {
                'title': 'Comprehensive Traffic Violation Analysis Report',
                'data': pd.DataFrame(),
                'sections': {}
            }
        
        df = pd.DataFrame(violations)
        
        # Generate all sub-reports
        daily_summary = self.generate_daily_summary_report()
        trend_analysis = self.generate_trend_analysis_report(30)
        vehicle_breakdown = self.generate_vehicle_breakdown_report()
        hotspot_analysis = self.generate_hotspot_analysis_report()
        time_patterns = self.generate_time_pattern_report()
        violation_history = self.generate_violation_history_report()
        
        # Executive summary
        executive_summary = {
            'report_period': f"{(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            'total_violations': len(df),
            'unique_vehicles': df['license_plate'].nunique() if 'license_plate' in df.columns else 0,
            'most_common_violation': df['violation_type'].mode().iloc[0] if not df.empty else 'None',
            'most_problematic_vehicle_type': df['vehicle_type'].mode().iloc[0] if not df.empty else 'None',
            'peak_violation_hour': df['timestamp'].dt.hour.mode().iloc[0] if not df.empty else 0,
            'average_daily_violations': len(df) / 30,
            'critical_hotspots': len(hotspot_analysis['critical_hotspots']) if not hotspot_analysis['critical_hotspots'].empty else 0
        }
        
        return {
            'title': 'Comprehensive Traffic Violation Analysis Report',
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'executive_summary': executive_summary,
            'data': df,
            'sections': {
                'daily_summary': daily_summary,
                'trend_analysis': trend_analysis,
                'vehicle_breakdown': vehicle_breakdown,
                'hotspot_analysis': hotspot_analysis,
                'time_patterns': time_patterns,
                'violation_history': violation_history
            }
        }
    
    def export_to_csv(self, report_data: Dict, report_type: str) -> bytes:
        """Export report data to CSV format"""
        output = io.StringIO()
        
        # Write report header
        output.write(f"{report_data['title']}\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write main data
        if not report_data['data'].empty:
            report_data['data'].to_csv(output, index=False)
            output.write("\n")
        
        # Write summary statistics
        if 'summary' in report_data:
            output.write("SUMMARY STATISTICS\n")
            for key, value in report_data['summary'].items():
                output.write(f"{key},{value}\n")
            output.write("\n")
        
        # Write additional sections based on report type
        if report_type == 'trend_analysis' and 'daily_trends' in report_data:
            output.write("DAILY TRENDS\n")
            report_data['daily_trends'].to_csv(output)
            output.write("\n")
        
        if report_type == 'hotspot_analysis' and 'critical_hotspots' in report_data:
            if not report_data['critical_hotspots'].empty:
                output.write("CRITICAL HOTSPOTS\n")
                report_data['critical_hotspots'].to_csv(output)
                output.write("\n")
        
        return output.getvalue().encode('utf-8')
    
    def export_to_excel(self, report_data: Dict, report_type: str) -> bytes:
        """Export report data to Excel format"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'bg_color': '#D9E2F3'
            })
            
            # Main data sheet
            if not report_data['data'].empty:
                report_data['data'].to_excel(writer, sheet_name='Main_Data', index=False)
                worksheet = writer.sheets['Main_Data']
                
                # Format headers
                for col_num, value in enumerate(report_data['data'].columns.values):
                    worksheet.write(0, col_num, value, header_format)
            
            # Summary sheet
            if 'summary' in report_data:
                summary_df = pd.DataFrame(list(report_data['summary'].items()), 
                                        columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                summary_worksheet = writer.sheets['Summary']
                summary_worksheet.write(0, 0, 'Metric', header_format)
                summary_worksheet.write(0, 1, 'Value', header_format)
            
            # Additional sheets based on report type
            if report_type == 'trend_analysis':
                if 'daily_trends' in report_data and not report_data['daily_trends'].empty:
                    report_data['daily_trends'].to_excel(writer, sheet_name='Daily_Trends')
                if 'weekly_trends' in report_data and not report_data['weekly_trends'].empty:
                    report_data['weekly_trends'].to_excel(writer, sheet_name='Weekly_Trends', index=False)
            
            elif report_type == 'vehicle_breakdown':
                if 'breakdown' in report_data and not report_data['breakdown'].empty:
                    report_data['breakdown'].to_excel(writer, sheet_name='Vehicle_Breakdown')
                if 'violation_matrix' in report_data and not report_data['violation_matrix'].empty:
                    report_data['violation_matrix'].to_excel(writer, sheet_name='Violation_Matrix')
            
            elif report_type == 'hotspot_analysis':
                if 'critical_hotspots' in report_data and not report_data['critical_hotspots'].empty:
                    report_data['critical_hotspots'].to_excel(writer, sheet_name='Critical_Hotspots')
            
            elif report_type == 'time_pattern':
                if 'hourly_violation_types' in report_data and not report_data['hourly_violation_types'].empty:
                    report_data['hourly_violation_types'].to_excel(writer, sheet_name='Hourly_Patterns')
                if 'daily_violation_types' in report_data and not report_data['daily_violation_types'].empty:
                    report_data['daily_violation_types'].to_excel(writer, sheet_name='Daily_Patterns')
            
            elif report_type == 'violation_history':
                if 'vehicle_history' in report_data and not report_data['vehicle_history'].empty:
                    report_data['vehicle_history'].to_excel(writer, sheet_name='Vehicle_History')
                if 'repeat_offenders' in report_data and not report_data['repeat_offenders'].empty:
                    report_data['repeat_offenders'].to_excel(writer, sheet_name='Repeat_Offenders')
        
        output.seek(0)
        return output.getvalue()
    
    def get_available_reports(self) -> Dict[str, str]:
        """Get list of available report types"""
        return self.report_types