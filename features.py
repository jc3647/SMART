
class FoodItem:
    def __init__(self, itemName, ifSolid, softness, price, salt_content, healthiness_index, sugar_content, protein, calories):
        self.itemName = itemName
        self.ifSolid = ifSolid
        self.softness = softness
        self.price = price
        self.salt_content = salt_content
        self.healthiness_index = healthiness_index
        self.sugar_content = sugar_content
        self.protein = protein
        self.calories = calories

    def display_details(self):
        print(f"Item Name: {self.itemName}, {'Solid' if self.ifSolid else 'Liquid'}, Softness: {self.softness}, Price: ${self.price}, Salt Content: {self.salt_content}, Healthiness Index: {self.healthiness_index}, Sugar Content: {self.sugar_content}, Protein: {self.protein}, Calories: {self.calories}")

    def normalize_data(self, min_values, max_values):
        self.softness = (self.softness - min_values['softness']) / (max_values['softness'] - min_values['softness'])
        self.price = (self.price - min_values['price']) / (max_values['price'] - min_values['price'])
        self.salt_content = (self.salt_content - min_values['salt_content']) / (max_values['salt_content'] - min_values['salt_content'])
        self.healthiness_index = (self.healthiness_index - min_values['healthiness_index']) / (max_values['healthiness_index'] - min_values['healthiness_index'])
        self.sugar_content = (self.sugar_content - min_values['sugar_content']) / (max_values['sugar_content'] - min_values['sugar_content'])
        self.protein = (self.protein - min_values['protein']) / (max_values['protein'] - min_values['protein'])
        self.calories = (self.calories - min_values['calories']) / (max_values['calories'] - min_values['calories'])

food_items = [
    FoodItem("Regular Potato Chips (Lays)", 1, 5.8, 2.00, 170, 52, 1, 2, 160),
    FoodItem("Barbeque Chips", 1, 6, 2.50, 150, 53, 2, 2, 160),
    FoodItem("Sour Cream Chips", 1, 5.6, 2.50, 200, 53, 2, 2, 160),
    FoodItem("Flaming Hot Cheetos", 1, 6.2, 3.00, 250, 21, 0, 1, 170),
    FoodItem("Coca-Cola", 0, 1, 1.50, 45, 1, 39, 0, 140),
    FoodItem("Coffee", 0, 1, 3.00, 120, 1, 12, 8, 140),
    FoodItem("Orange Juice", 0, 1, 2.00, 3, 79, 28, 3, 153),
    FoodItem("Water", 0, 1, 1.00, 0, 100, 0, 0, 0),
    FoodItem("Hershey's Chocolate", 1, 4.2, 1.50, 30, 14, 20, 4, 210),
    FoodItem("Reese's Peanut Butter Cups", 1, 3.2, 1.50, 150, 28, 22, 5, 210),
    FoodItem("Skittles", 1, 6.2, 1.50, 20, 1, 45, 0, 250),
    FoodItem("Kit Kat", 1, 4.4, 1.50, 30, 6, 22, 3, 210),
    FoodItem("Apple", 1, 6.4, 0.50, 2, 92, 19, 1, 95),
    FoodItem("Banana", 1, 2.4, 0.50, 0, 79, 17, 1, 120),
    FoodItem("Pear", 1, 5.8, 0.75, 0, 82, 16, 1, 100),
    FoodItem("Orange", 1, 3.4, 0.75, 0, 100, 14, 1, 70),
    FoodItem("Popcorn", 1, 5, 1.50, 820, 61, 0, 2, 425),
    FoodItem("Ramen", 1, 2.5, 0.50, 1590, 1, 0, 9, 380),
    FoodItem("Mac and Cheese", 1, 2.7, 1.50, 490, 11, 5, 7, 220),
    FoodItem("Chicken Noodle Soup", 1, 2.2, 3.00, 2225, 21, 1, 8, 150)
]

min_values = {
    "softness": float('inf'),
    "price": float('inf'),
    "salt_content": float('inf'),
    "healthiness_index": float('inf'),
    "sugar_content": float('inf'),
    "protein": float('inf'),
    "calories": float('inf')
}
max_values = {
    "softness": float('-inf'),
    "price": float('-inf'),
    "salt_content": float('-inf'),
    "healthiness_index": float('-inf'),
    "sugar_content": float('-inf'),
    "protein": float('-inf'),
    "calories": float('-inf')
}

# get ranges for normalization
for food in food_items:
    for key in min_values.keys():
        min_values[key] = min(min_values[key], getattr(food, key))
        max_values[key] = max(max_values[key], getattr(food, key))

for food in food_items:
    food.normalize_data(min_values, max_values)
    food.display_details()


