show databases; 
use sarawak_handicrat; 
CREATE TABLE Categories (
    CategoryID INT PRIMARY KEY AUTO_INCREMENT,
    CategoryName VARCHAR(255) NOT NULL,
    ParentCategoryID INT NULL,
    FOREIGN KEY (ParentCategoryID) REFERENCES Categories(CategoryID)
);

-- Create Products Table
CREATE TABLE Products (
    ProductID INT PRIMARY KEY AUTO_INCREMENT,
    ProductName VARCHAR(255) NOT NULL,
    Description TEXT,
    Price DECIMAL(10,2) NOT NULL,
    CategoryID INT,
    ImageURL VARCHAR(255),
    FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID)
);

-- Insert Data into Categories
INSERT INTO Categories (CategoryID, CategoryName, ParentCategoryID) VALUES
(1, 'Bag', NULL),
(2, 'Beadwork', NULL),
(3, 'Jewellery', NULL),
(4, 'Beads Jewellery', 3),
(5, 'Necklace', 3),
(6, 'Painting', NULL),
(7, 'Fabric Weaving', NULL),
(8, 'Weaving & Basketry', NULL),
(9, 'Wood Carving', NULL),
(10, 'Traditional Costume', NULL);

-- Insert Data into Products
INSERT INTO Products (ProductName, Description, Price, CategoryID, ImageURL) VALUES
('Antique Pendant Necklace', 'Silver antique pendant', 85.00, 5, '/images/antique_pendant_necklace.jpg'),
('Handwoven Penan Basket', 'Traditional Penan basket', 40.00, 8, '/images/handwoven_penan_basket.jpg'),
('Beaded Bracelet', 'Handmade colorful beadwork', 25.00, 4, '/images/beaded_bracelet.jpg'),
('Sarawak Batik Scarf', 'Traditional batik scarf', 60.00, 7, '/images/sarawak_batik_scarf.jpg'),
('Pua Kumbu Fabric', 'Traditional woven textile', 150.00, 7, '/images/pua_kumbu_fabric.jpg'),
('Hand-Carved Wooden Mask', 'Unique tribal wood carving', 200.00, 9, '/images/hand_carved_wooden_mask.jpg'),
('Rattan Handbag', 'Handmade woven rattan bag', 70.00, 1, '/images/rattan_handbag.jpg'),
('Bamboo Wall Art', 'Artistic bamboo carving', 120.00, 6, '/images/bamboo_wall_art.jpg'),
('Iban Warrior Costume', 'Traditional Iban warrior set', 300.00, 10, '/images/iban_warrior_costume.jpg'),
('Borneo Beaded Earrings', 'Handmade tribal earrings', 35.00, 4, '/images/borneo_beaded_earrings.jpg');