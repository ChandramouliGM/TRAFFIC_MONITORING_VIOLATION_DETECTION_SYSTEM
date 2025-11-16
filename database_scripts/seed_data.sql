-- Seed Data for Traffic Violation Detection System
-- Created: 2025-09-14
-- Description: Sample data for testing the traffic violation detection system

-- Insert sample violation data for testing
INSERT INTO violations (
    timestamp, frame_number, vehicle_type, license_plate, 
    violation_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height
) VALUES 
    ('2025-09-14 10:30:00', 150, 'car', 'ABC123', 'Red Light Violation', 0.95, 100, 50, 200, 150),
    ('2025-09-14 11:15:30', 200, 'truck', 'XYZ789', 'Speed Limit Violation', 0.88, 150, 75, 250, 180),
    ('2025-09-14 12:45:15', 75, 'car', 'DEF456', 'Wrong Lane Violation', 0.92, 80, 60, 180, 140),
    ('2025-09-14 13:20:45', 300, 'motorcycle', 'GHI789', 'No Helmet Violation', 0.87, 120, 90, 150, 120),
    ('2025-09-14 14:10:20', 180, 'car', 'JKL012', 'Stop Sign Violation', 0.93, 200, 100, 220, 160)
ON CONFLICT DO NOTHING;

-- Note: Admin user is created in create_tables.sql
-- Username: admin
-- Password: admin123 (bcrypt hashed)