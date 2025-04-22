show databases; 
use comments; 
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL, -- Allows anonymous comments if needed
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE comments_locations (
    id SERIAL PRIMARY KEY,
    comment_id INT REFERENCES comments(id) ON DELETE CASCADE,
    location_id INT REFERENCES locations(id) ON DELETE CASCADE
);

INSERT INTO users (name, email, created_at) VALUES
('Alice Johnson', 'alice@example.com', NOW()),
('Bob Smith', 'bob@example.com', NOW()),
('Charlie Brown', 'charlie@example.com', NOW()),
('David Lee', 'david@example.com', NOW()),
('Emma Watson', 'emma@example.com', NOW()),
('Frank Ocean', 'frank@example.com', NOW()),
('Grace Hopper', 'grace@example.com', NOW()),
('Hannah Baker', 'hannah@example.com', NOW()),
('Isaac Newton', 'isaac@example.com', NOW()),
('Jack Daniels', 'jack@example.com', NOW());

INSERT INTO comments (user_id, text, created_at) VALUES
(1, 'Great service! Really enjoyed my experience.', NOW()),
(2, 'Could use some improvements in the UI.', NOW()),
(3, 'Love the new updates, keep it up!', NOW()),
(4, 'Had some issues with navigation, please fix.', NOW()),
(5, 'Excellent customer support, very responsive.', NOW()),
(6, 'The performance has improved a lot, thanks!', NOW()),
(7, 'Found a bug in the checkout process.', NOW()),
(8, 'Would love a dark mode feature.', NOW()),
(9, 'Amazing work, looking forward to new features.', NOW()),
(10, 'Some pages take longer to load, need optimization.', NOW());

INSERT INTO locations (name, description, created_at) VALUES
('Central Park', 'A large public park in New York City.', NOW()),
('Eiffel Tower', 'Iconic Paris landmark with stunning views.', NOW()),
('Great Wall of China', 'Ancient wall stretching across China.', NOW()),
('Sydney Opera House', 'Famous performing arts center in Australia.', NOW()),
('Grand Canyon', 'Breathtaking canyon in Arizona.', NOW()),
('Machu Picchu', 'Historic Incan city in Peru.', NOW()),
('Times Square', 'Vibrant tourist attraction in NYC.', NOW()),
('Mount Everest', 'Tallest mountain in the world.', NOW()),
('Taj Mahal', 'Beautiful mausoleum in India.', NOW()),
('Colosseum', 'Ancient Roman amphitheater in Italy.', NOW());

INSERT INTO comments_locations (comment_id, location_id) VALUES
(1, 1),  -- Comment 1 about Central Park
(2, 2),  -- Comment 2 about Eiffel Tower
(3, 3),  -- Comment 3 about Great Wall of China
(4, 4),  -- Comment 4 about Sydney Opera House
(5, 5),  -- Comment 5 about Grand Canyon
(6, 6),  -- Comment 6 about Machu Picchu
(7, 7),  -- Comment 7 about Times Square
(8, 8),  -- Comment 8 about Mount Everest
(9, 9),  -- Comment 9 about Taj Mahal
(10, 10); -- Comment 10 about Colosseum; 