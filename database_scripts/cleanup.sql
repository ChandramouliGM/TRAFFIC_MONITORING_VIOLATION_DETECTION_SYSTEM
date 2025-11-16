-- Cleanup Script for Traffic Violation Detection System Database
-- Created: 2025-09-14
-- Description: Scripts to clean up or reset database tables

-- WARNING: These commands will delete all data. Use with caution!

-- Clear all violations (for testing purposes)
-- TRUNCATE violations RESTART IDENTITY CASCADE;

-- Clear all users except admin (for testing purposes)
-- DELETE FROM users WHERE username != 'admin';

-- Reset violations table sequence
-- ALTER SEQUENCE violations_id_seq RESTART WITH 1;

-- Reset users table sequence  
-- ALTER SEQUENCE users_id_seq RESTART WITH 1;

-- Drop all tables (complete reset - DANGER!)
-- DROP TABLE IF EXISTS violations CASCADE;
-- DROP TABLE IF EXISTS users CASCADE;

-- Note: Uncomment the lines above only when you need to perform cleanup
-- All commands are commented for safety