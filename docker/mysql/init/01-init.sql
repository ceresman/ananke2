CREATE DATABASE IF NOT EXISTS ananke;
CREATE USER IF NOT EXISTS 'ananke'@'%' IDENTIFIED WITH mysql_native_password BY 'anankepass';
GRANT ALL PRIVILEGES ON ananke.* TO 'ananke'@'%';
FLUSH PRIVILEGES;
