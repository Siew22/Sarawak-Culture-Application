show databases; 
use shared; 
CREATE TABLE users (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    email        VARCHAR(255) NOT NULL UNIQUE, 
    password_hash TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    id          INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id     INT UNSIGNED NOT NULL,
    device_info TEXT DEFAULT NULL,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE auth_tokens (
    id             INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id        INT UNSIGNED NOT NULL,
    access_token   TEXT NOT NULL,
    refresh_token  TEXT NOT NULL,
    expires_at     DATETIME NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (access_token(255)),
    UNIQUE (refresh_token(255))
);

INSERT INTO users (email, password_hash) VALUES
('user1@example.com', 'bcrypt_hash_1'),
('user2@example.com', 'bcrypt_hash_2'),
('user3@example.com', 'bcrypt_hash_3'),
('user4@example.com', 'bcrypt_hash_4'),
('user5@example.com', 'bcrypt_hash_5'),
('user6@example.com', 'bcrypt_hash_6'),
('user7@example.com', 'bcrypt_hash_7'),
('user8@example.com', 'bcrypt_hash_8'),
('user9@example.com', 'bcrypt_hash_9'),
('user10@example.com', 'bcrypt_hash_10');

INSERT INTO sessions (user_id, device_info) VALUES
(1, 'Windows - Chrome'),
(2, 'MacOS - Safari'),
(3, 'Android - Firefox'),
(4, 'iPhone - Safari'),
(5, 'Linux - Chromium'),
(6, 'Windows - Edge'),
(7, 'MacOS - Chrome'),
(8, 'Android - Chrome'),
(9, 'iPhone - Edge'),
(10, 'Linux - Firefox');

INSERT INTO auth_tokens (user_id, access_token, refresh_token, expires_at) VALUES
(1, 'token_access_1', 'token_refresh_1', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(2, 'token_access_2', 'token_refresh_2', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(3, 'token_access_3', 'token_refresh_3', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(4, 'token_access_4', 'token_refresh_4', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(5, 'token_access_5', 'token_refresh_5', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(6, 'token_access_6', 'token_refresh_6', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(7, 'token_access_7', 'token_refresh_7', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(8, 'token_access_8', 'token_refresh_8', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(9, 'token_access_9', 'token_refresh_9', DATE_ADD(NOW(), INTERVAL 7 DAY)),
(10, 'token_access_10', 'token_refresh_10', DATE_ADD(NOW(), INTERVAL 7 DAY));

select * from users; 