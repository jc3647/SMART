
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
# for food in food_items:
#     for key in min_values.keys():
#         min_values[key] = min(min_values[key], getattr(food, key))
#         max_values[key] = max(max_values[key], getattr(food, key))

# for food in food_items:
#     food.normalize_data(min_values, max_values)
#     food.display_details()


