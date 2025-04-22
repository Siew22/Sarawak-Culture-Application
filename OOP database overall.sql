show databases; 
create database tourist;
Use tourist; 
CREATE TABLE `users` (
  `id` integer PRIMARY KEY,
  `name` varchar(255),
  `email` varchar(255) UNIQUE,
  `password` varchar(255),
  `role` varchar(255),
  `created_at` timestamp
);

CREATE TABLE `attractions` (
  `id` integer PRIMARY KEY,
  `name` varchar(255),
  `location` text,
  `type` varchar(255),
  `description` text,
  `created_at` timestamp
);

CREATE TABLE `tourism_business` (
  `id` integer PRIMARY KEY,
  `name` varchar(255),
  `type` varchar(255),
  `location` text,
  `owner_id` integer NOT NULL,
  `created_at` timestamp
);

CREATE TABLE `bookings` (
  `id` integer PRIMARY KEY,
  `user_id` integer NOT NULL,
  `business_id` integer NOT NULL , 
  date DATE NOT NULL, 
  status ENUM('Pending', 'Confirmed', 'Cancelled') NOT NULL, 
  created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE `payments` (
  `id` integer PRIMARY KEY,
  `booking_id` integer NOT NULL,
  `amount` decimal(10,2),
  `payment_method` varchar(255),
  `status` varchar(255),
  `transaction_date` timestamp
);

CREATE TABLE `itineraries` (
  `id` integer PRIMARY KEY,
  `user_id` integer NOT NULL,
  `name` varchar(255),
  `created_date` timestamp
);

CREATE TABLE `reviews` (
  `id` integer PRIMARY KEY,
  `user_id` integer NOT NULL,
  `attraction_id` integer NOT NULL,
  `rating` integer,
  `comment` text,
  `created_at` timestamp
);

CREATE TABLE `recommendations` (
  `id` integer PRIMARY KEY,
  `user_id` integer NOT NULL,
  `suggested_attractions` text,
  `generated_at` timestamp
);

ALTER TABLE `bookings` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `bookings` ADD FOREIGN KEY (`business_id`) REFERENCES `tourism_business` (`id`);

ALTER TABLE `payments` ADD FOREIGN KEY (`booking_id`) REFERENCES `bookings` (`id`);

ALTER TABLE `itineraries` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `reviews` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `reviews` ADD FOREIGN KEY (`attraction_id`) REFERENCES `attractions` (`id`);

ALTER TABLE `recommendations` ADD FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

ALTER TABLE `tourism_business` ADD FOREIGN KEY (`owner_id`) REFERENCES `users` (`id`);



INSERT INTO users (id, name, email, password, role, created_at) VALUES
(1, 'Alice Johnson', 'alice@example.com', 'password123', 'Tourist', NOW()),
(2, 'Bob Smith', 'bob@example.com', 'securepass', 'Business', NOW()),
(3, 'Charlie Lee', 'charlie@example.com', 'adminpass', 'Admin', NOW()),
(4, 'David Tan', 'david@example.com', 'pass456', 'Tourist', NOW()),
(5, 'Emma Wong', 'emma@example.com', 'qwerty', 'Business', NOW()),
(6, 'Faisal Rahman', 'faisal@example.com', 'abc123', 'Tourist', NOW()),
(7, 'Grace Liu', 'grace@example.com', 'pass789', 'Business', NOW()),
(8, 'Henry Lim', 'henry@example.com', 'henrypass', 'Admin', NOW()),
(9, 'Ivy Chen', 'ivy@example.com', 'welcome1', 'Tourist', NOW()),
(10, 'Jacky Ho', 'jacky@example.com', 'jackypass', 'Business', NOW());

INSERT INTO attractions (id, name, location, type, description, created_at) VALUES
(1, 'Rainforest World Music Festival', 'Sarawak Cultural Village', 'Cultural', 'Annual world music festival in Sarawak', NOW()),
(2, 'Mulu Caves', 'Gunung Mulu National Park', 'Nature', 'Largest cave chamber in the world', NOW()),
(3, 'Sarawak Museum', 'Kuching', 'Heritage', 'Oldest museum in Borneo', NOW()),
(4, 'Bako National Park', 'Kuching', 'Nature', 'Famous for its wildlife and beaches', NOW()),
(5, 'Semenggoh Wildlife Centre', 'Sarawak', 'Nature', 'Orangutan rehabilitation center', NOW()),
(6, 'Sarawak River Cruise', 'Kuching', 'Cultural', 'Scenic cruise along the Sarawak River', NOW()),
(7, 'Damai Beach', 'Santubong', 'Nature', 'Beautiful sandy beach', NOW()),
(8, 'The Waterfront', 'Kuching', 'Heritage', 'Popular tourist area with restaurants', NOW()),
(9, 'Gunung Gading National Park', 'Sarawak', 'Nature', 'Home of the world’s largest flower, Rafflesia', NOW()),
(10, 'Satok Weekend Market', 'Kuching', 'Cultural', 'Traditional market with local goods', NOW());

INSERT INTO tourism_business (id, name, type, location, owner_id, created_at) VALUES
(1, 'Borneo Adventure Tours', 'Tour Operator', 'Kuching', 2, NOW()),
(2, 'Sarawak Grand Hotel', 'Hotel', 'Miri', 5, NOW()),
(3, 'Heritage Cafe', 'Restaurant', 'Sarawak Museum Area', 5, NOW()),
(4, 'Jungle Trek Tours', 'Tour Operator', 'Mulu Caves', 7, NOW()),
(5, 'Sunset Resort', 'Hotel', 'Damai Beach', 7, NOW()),
(6, 'River Breeze Restaurant', 'Restaurant', 'Waterfront, Kuching', 10, NOW()),
(7, 'Eco Sarawak Travel', 'Tour Operator', 'Gunung Gading National Park', 10, NOW()),
(8, 'Bako Lodge', 'Hotel', 'Bako National Park', 2, NOW()),
(9, 'Wildlife Explorer', 'Tour Operator', 'Semenggoh Wildlife Centre', 5, NOW()),
(10, 'Sunset Boat Cruise', 'Tour Operator', 'Sarawak River', 7, NOW());

INSERT INTO bookings (id, user_id, business_id, date, status, created_at) VALUES
(1, 1, 1, '2025-03-10 14:00:00', 'Confirmed', NOW()),
(2, 1, 2, '2025-03-15 18:00:00', 'Pending', NOW()),
(3, 2, 3, '2025-03-20 12:30:00', 'Cancelled', NOW()),
(4, 3, 4, '2025-03-22 10:00:00', 'Confirmed', NOW()),
(5, 4, 5, '2025-03-25 16:00:00', 'Confirmed', NOW()),
(6, 5, 6, '2025-03-30 19:00:00', 'Pending', NOW()),
(7, 6, 7, '2025-04-01 08:00:00', 'Confirmed', NOW()),
(8, 7, 8, '2025-04-05 14:00:00', 'Cancelled', NOW()),
(9, 8, 9, '2025-04-07 11:00:00', 'Confirmed', NOW()),
(10, 9, 10, '2025-04-10 17:30:00', 'Pending', NOW());

INSERT INTO payments (id, booking_id, amount, payment_method, status, transaction_date) VALUES
(1, 1, 150.00, 'S Pay Global', 'Paid', NOW()),
(2, 2, 250.00, 'Credit Card', 'Pending', NOW()),
(3, 3, 50.00, 'PayPal', 'Refunded', NOW()),
(4, 4, 300.00, 'Credit Card', 'Paid', NOW()),
(5, 5, 400.00, 'S Pay Global', 'Paid', NOW()),
(6, 6, 120.00, 'PayPal', 'Pending', NOW()),
(7, 7, 220.00, 'S Pay Global', 'Paid', NOW()),
(8, 8, 350.00, 'Credit Card', 'Refunded', NOW()),
(9, 9, 180.00, 'S Pay Global', 'Paid', NOW()),
(10, 10, 90.00, 'PayPal', 'Pending', NOW());

INSERT INTO reviews (id, user_id, attraction_id, rating, comment, created_at) VALUES
(1, 1, 1, 5, 'Amazing festival! Highly recommend.', NOW()),
(2, 1, 2, 4, 'Breathtaking caves, but be ready for a long walk.', NOW()),
(3, 2, 3, 3, 'Interesting history, but small museum.', NOW()),
(4, 3, 4, 5, 'Loved the wildlife and scenery!', NOW()),
(5, 4, 5, 4, 'Great place to see orangutans up close.', NOW()),
(6, 5, 6, 5, 'Beautiful cruise with stunning views.', NOW()),
(7, 6, 7, 4, 'Perfect beach for relaxation.', NOW()),
(8, 7, 8, 3, 'Nice spot but crowded in the evening.', NOW()),
(9, 8, 9, 5, 'Saw the Rafflesia flower in full bloom!', NOW()),
(10, 9, 10, 4, 'Great market for souvenirs.', NOW());

INSERT INTO itineraries (id, user_id, name, created_date) VALUES
(1, 1, 'Sarawak Adventure Trip', NOW()),
(2, 1, 'Nature Explorer Tour', NOW()),
(3, 2, 'Cultural Heritage Journey', NOW()),
(4, 3, 'Rainforest Escape', NOW()),
(5, 4, 'Wildlife and Conservation Tour', NOW()),
(6, 5, 'Beach and Island Getaway', NOW()),
(7, 6, 'City and Shopping Experience', NOW()),
(8, 7, 'Local Food and Market Tour', NOW()),
(9, 8, 'Hiking and Outdoor Adventures', NOW()),
(10, 9, 'Historical and Museum Tour', NOW());

INSERT INTO recommendations (id, user_id, suggested_attractions, generated_at) VALUES
(1, 1, 'Mulu Caves, Rainforest Festival, Sarawak River Cruise', NOW()),
(2, 2, 'Sarawak Museum, The Waterfront, Bako National Park', NOW()),
(3, 3, 'Gunung Gading National Park, Semenggoh Wildlife Centre', NOW()),
(4, 4, 'Rainforest Festival, Damai Beach, Sunset Boat Cruise', NOW()),
(5, 5, 'Sarawak River Cruise, Satok Weekend Market, The Waterfront', NOW()),
(6, 6, 'Mulu Caves, Bako National Park, Jungle Trek Tours', NOW()),
(7, 7, 'Sarawak Museum, Heritage Cafe, Sarawak Grand Hotel', NOW()),
(8, 8, 'Bako Lodge, Semenggoh Wildlife Centre, Wildlife Explorer', NOW()),
(9, 9, 'Gunung Gading National Park, Sunset Resort, Damai Beach', NOW()),
(10, 10, 'Sarawak Grand Hotel, River Breeze Restaurant, Heritage Cafe', NOW());

select * from users; 
select * from attractions; 
select * from tourism_business; 
select * from bookings; 
select * from payments; 
select * from reviews; 
select * from itineraries; 
select * from recommendations;