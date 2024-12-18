CREATE DATABASE IF NOT EXISTS ananke;
GRANT ALL PRIVILEGES ON ananke.* TO 'ananke'@'%';
ALTER USER 'ananke'@'%' IDENTIFIED WITH mysql_native_password BY 'anankepass';
FLUSH PRIVILEGES;
