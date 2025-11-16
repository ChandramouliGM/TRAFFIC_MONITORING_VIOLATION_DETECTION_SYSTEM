import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import streamlit as st
import bcrypt

class DatabaseManager:
    """Manages PostgreSQL database operations for traffic violations"""
    
    def __init__(self):
        """Initialize database connection using environment variables"""
        self.connection_params = {
            'host': os.getenv('PGHOST', 'localhost'),
            'port': os.getenv('PGPORT', '5432'),
            'database': os.getenv('PGDATABASE', 'traffic_violations'),
            'user': os.getenv('PGUSER', 'postgres'),
            'password': os.getenv('PGPASSWORD', 'pgsql@123$')
        }
        
        # Alternative: use DATABASE_URL if provided
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            self.database_url = database_url
        else:
            self.database_url = None
        
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        try:
            if self.database_url:
                conn = psycopg2.connect(self.database_url)
            else:
                conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            # Log server-side only, don't expose to unauthenticated users
            import logging
            logging.error(f"Database connection error: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = self.get_connection()
            if conn:
                conn.close()
                return True
            return False
        except Exception:
            return False
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # Create violations table with enhanced fields
            create_violations_query = """
            CREATE TABLE IF NOT EXISTS violations (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                frame_number INTEGER,
                vehicle_type VARCHAR(50),
                license_plate VARCHAR(20),
                violation_type VARCHAR(100),
                confidence FLOAT,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                speed_kph FLOAT DEFAULT 0,
                behavior_flags JSONB,
                plate_confidence FLOAT DEFAULT 0,
                bbox JSONB,
                features JSONB,
                alert_events JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_violations_license_plate ON violations(license_plate);
            CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
            """
            
            # Create users table with proper security
            create_users_query = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            """
            
            cursor.execute(create_violations_query)
            cursor.execute(create_users_query)
            conn.commit()
            
            # Seed admin user with hashed password
            self._seed_admin_user(cursor, conn)
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            import logging
            logging.error(f"Database initialization error: {str(e)}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _seed_admin_user(self, cursor, conn):
        """Seed admin user only in development or with explicit env vars"""
        try:
            # Only seed in development or with explicit admin credentials
            is_dev = os.getenv('STREAMLIT_ENV') == 'development' or os.getenv('DEBUG') == 'true'
            admin_username = os.getenv('ADMIN_USERNAME')
            admin_password = os.getenv('ADMIN_PASSWORD')
            
            if not (is_dev or (admin_username and admin_password)):
                return  # Don't seed admin in production without explicit env vars
            
            # Use env vars if provided, otherwise dev defaults
            username = admin_username or 'admin'
            password = admin_password or 'admin123'
            
            # Check if admin user exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return  # Admin already exists
            
            # Create admin user with hashed password
            hashed_password = self._hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, password_hash, role) 
                VALUES (%s, %s, %s)
            """, (username, hashed_password, 'admin'))
            
            conn.commit()
            
            # Log creation (server-side only)
            import logging
            logging.info(f"Admin user '{username}' created successfully")
            
        except Exception as e:
            import logging
            logging.error(f"Error seeding admin user: {str(e)}")
            pass
    
    def insert_violation(self, violation_data: Dict) -> bool:
        """Insert a violation record into the database with enhanced fields"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()

            # Extract bbox coordinates (legacy format)
            bbox = violation_data.get('bbox', [0, 0, 0, 0])
            bbox_x, bbox_y, bbox_width, bbox_height = bbox if len(bbox) == 4 else [0, 0, 0, 0]

            # Prepare enhanced fields with proper JSON handling
            speed_kph = violation_data.get('speed', 0)
            behavior_flags = Json(violation_data.get('behavior_flags', {})) if violation_data.get('behavior_flags') else None
            plate_confidence = violation_data.get('plate_confidence', 0.0)
            bbox_json = Json(bbox) if bbox else None
            features = Json(violation_data.get('vehicle_features', {})) if violation_data.get('vehicle_features') else None
            alert_events = Json(violation_data.get('alert_events', [])) if violation_data.get('alert_events') else None

            # Try enhanced insert first, fallback to basic if columns do not exist
            primary_error = None
            try:
                insert_query = """
                INSERT INTO violations (
                    timestamp, frame_number, vehicle_type, license_plate,
                    violation_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height,
                    speed_kph, behavior_flags, plate_confidence, bbox, features, alert_events
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(insert_query, (
                    violation_data['timestamp'],
                    violation_data.get('frame_number'),
                    violation_data.get('vehicle_type'),
                    violation_data.get('license_plate'),
                    violation_data.get('violation_type'),
                    violation_data.get('confidence'),
                    bbox_x, bbox_y, bbox_width, bbox_height,
                    speed_kph, behavior_flags, plate_confidence, bbox_json, features, alert_events
                ))
            except Exception as exc:
                primary_error = exc
                if conn:
                    conn.rollback()
                basic_insert_query = """
                INSERT INTO violations (
                    timestamp, frame_number, vehicle_type, license_plate,
                    violation_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                try:
                    cursor.execute(basic_insert_query, (
                        violation_data['timestamp'],
                        violation_data.get('frame_number'),
                        violation_data.get('vehicle_type'),
                        violation_data.get('license_plate'),
                        violation_data.get('violation_type'),
                        violation_data.get('confidence'),
                        bbox_x, bbox_y, bbox_width, bbox_height
                    ))
                except Exception as fallback_error:
                    if conn:
                        conn.rollback()
                    if primary_error:
                        raise fallback_error from primary_error
                    raise

            if conn:
                conn.commit()
            return True

        except Exception as e:
            import logging
            logging.error(f"Error inserting violation: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def get_violations(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None,
                      violation_type: Optional[str] = None) -> List[Dict]:
        """Retrieve violations from database with optional filters"""
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query with filters
            query = """
            SELECT id, timestamp, frame_number, vehicle_type, license_plate, 
                   violation_type, confidence, created_at
            FROM violations 
            WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            if violation_type:
                query += " AND violation_type = %s"
                params.append(violation_type)
            
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            st.error(f"Error retrieving violations: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get violation statistics for dashboard"""
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            stats = {}
            
            # Total violations
            cursor.execute("SELECT COUNT(*) as total FROM violations")
            result = cursor.fetchone()
            stats['total_violations'] = result['total'] if result else 0
            
            # Unique vehicles
            cursor.execute("SELECT COUNT(DISTINCT license_plate) as unique FROM violations WHERE license_plate IS NOT NULL")
            result = cursor.fetchone()
            stats['unique_vehicles'] = result['unique'] if result else 0
            
            # Today's violations
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) as today FROM violations WHERE DATE(timestamp) = %s", (today,))
            result = cursor.fetchone()
            stats['today_violations'] = result['today'] if result else 0
            
            # This week's violations
            week_start = today - timedelta(days=today.weekday())
            cursor.execute("SELECT COUNT(*) as week FROM violations WHERE DATE(timestamp) >= %s", (week_start,))
            result = cursor.fetchone()
            stats['week_violations'] = result['week'] if result else 0
            
            # Violations by type
            cursor.execute("""
                SELECT violation_type, COUNT(*) as count 
                FROM violations 
                WHERE violation_type IS NOT NULL
                GROUP BY violation_type
                ORDER BY count DESC
            """)
            violation_types = {row['violation_type']: row['count'] for row in cursor.fetchall()}
            stats['violation_types'] = violation_types
            
            # Violations by vehicle type
            cursor.execute("""
                SELECT vehicle_type, COUNT(*) as count 
                FROM violations 
                WHERE vehicle_type IS NOT NULL
                GROUP BY vehicle_type
                ORDER BY count DESC
            """)
            vehicle_types = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}
            stats['vehicle_types'] = vehicle_types
            
            # Daily trend (last 7 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM violations 
                WHERE timestamp >= %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (datetime.now() - timedelta(days=7),))
            
            daily_trend = [{'date': row['date'].strftime('%Y-%m-%d'), 'violations': row['count']} 
                          for row in cursor.fetchall()]
            stats['daily_trend'] = daily_trend
            
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
            return {}
    
    def clear_violations(self) -> bool:
        """Clear all violation records (for testing purposes)"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM violations")
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error clearing violations: {str(e)}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user login with secure password verification"""
        try:
            conn = self.get_connection()
            if not conn:
                return None
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # First, get user by username only to retrieve password hash
            query = "SELECT id, username, password_hash, role, created_at FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            
            user = cursor.fetchone()
            
            if user and self._verify_password(password, user['password_hash']):
                # Password verified, update last login
                update_query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s"
                cursor.execute(update_query, (user['id'],))
                conn.commit()
                
                # Return user data without password hash
                user_data = {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'created_at': user['created_at']
                }
                
                cursor.close()
                conn.close()
                return user_data
            
            cursor.close()
            conn.close()
            return None
            
        except Exception as e:
            # Generic error message to avoid leaking internals
            import logging
            logging.error(f"Authentication error: {str(e)}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            conn = self.get_connection()
            if not conn:
                return None
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT id, username, role, created_at, last_login FROM users WHERE id = %s"
            cursor.execute(query, (user_id,))
            
            user = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(user) if user else None
            
        except Exception as e:
            st.error(f"Error fetching user: {str(e)}")
            return None


