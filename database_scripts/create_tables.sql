-- Traffic Violation Detection System Database Schema
-- Created: 2025-09-14
-- Description: Complete database schema for traffic violation detection system with secure authentication

-- Create violations table for storing detected traffic violations
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create users table for secure admin authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_license_plate ON violations(license_plate);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Insert default admin user (password: admin123)
-- Note: Password hash generated using bcrypt with salt rounds=12
INSERT INTO users (username, password_hash, role) 
VALUES ('admin', '$2b$12$kolEiyvqilc2Y7HJ0CDBbOcX3/kxfK5azVOy2ngmJUvtnvnM3d.cO', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Grant necessary permissions (adjust schema name if needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;