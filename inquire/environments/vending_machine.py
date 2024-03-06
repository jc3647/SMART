
import numpy as np
from typing import Union
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
# from inquire.utils.features import FoodItem
# from utils.features import FoodItem

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
class Trajectory:
    def __init__(self, states: list, actions: list, phi: Union[list, np.ndarray]):
        self.phi = phi
        self.states = states
        self.actions = actions


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


class VendingMachine():
    def __init__(self, decay_rate=0.99, food_items=food_items):
        self.food_items = food_items
        self.num_items = len(self.food_items)
        self.feature_dimensions = 8
        self.num_components = 3
        self.gmm = GaussianMixture(n_components=self.num_components)#, covariance_type='full', max_iter=100, n_init=1, init_params='kmeans')
        self.user_preferences = np.ones(self.num_items) / self.num_items
        self.decay_rate = decay_rate
        # normalize food items feature weights
        self.min_values = {
            "softness": float('inf'),
            "price": float('inf'),
            "salt_content": float('inf'),
            "healthiness_index": float('inf'),
            "sugar_content": float('inf'),
            "protein": float('inf'),
            "calories": float('inf')
        }
        self.max_values = {
            "softness": float('-inf'),
            "price": float('-inf'),
            "salt_content": float('-inf'),
            "healthiness_index": float('-inf'),
            "sugar_content": float('-inf'),
            "protein": float('-inf'),
            "calories": float('-inf')
        }
        for food in self.food_items:
            for key in self.min_values.keys():
                self.min_values[key] = min(self.min_values[key], getattr(food, key))
                self.max_values[key] = max(self.max_values[key], getattr(food, key))

        for food in self.food_items:
            food.normalize_data(self.min_values, self.max_values)

    def train_gmm(self):
        features = np.array([[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
        self.gmm.fit(features)

    def generate_random_food(self):
        return np.random.choice(self.food_items)
    
    def provide_demonstration(self, food):
        for item in self.food_items:
            if item.itemName == food:
                return item
            
    def update_preferences_by_cosine_similarity(self, feedback, modality="B", suggestion=None):
        # demonstration update
        if modality == "D":
            weights = np.array([
                feedback.ifSolid, feedback.softness, feedback.price, feedback.salt_content, feedback.healthiness_index, feedback.sugar_content, feedback.protein, feedback.calories
            ])
            # compute cosine similarity score between chosen item and everything else
            similarities = cosine_similarity([weights], [[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
            self.user_preferences += self.decay_rate * similarities[0]

        # correction update
        elif modality == "C":
            print("suggested: ", suggestion.itemName)
            print("feedback: ", feedback.itemName)
            suggestion_weights = np.array([
                suggestion.ifSolid, suggestion.softness, suggestion.price, suggestion.salt_content, suggestion.healthiness_index, suggestion.sugar_content, suggestion.protein, suggestion.calories
            ])
            preferred_weights = np.array([
                feedback.ifSolid, feedback.softness, feedback.price, feedback.salt_content, feedback.healthiness_index, feedback.sugar_content, feedback.protein, feedback.calories
            ])
            similarity_suggested_to_preferred = cosine_similarity([suggestion_weights], [preferred_weights])[0][0]
            print("similarity_suggested_to_preferred: ", similarity_suggested_to_preferred) 

            for i, item in enumerate(self.food_items):
                item_weights = np.array([
                    item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories
                ])
                similarity_suggested_to_item = cosine_similarity([suggestion_weights], [item_weights])[0][0]
                similarity_preferred_to_item = cosine_similarity([preferred_weights], [item_weights])[0][0]
                similarity_difference = similarity_preferred_to_item - similarity_suggested_to_item

                if similarity_difference != 0:
                    # self.user_preferences[i] += similarity_difference * (1 - similarity_suggested_to_preferred) * self.decay_rate
                    self.user_preferences[i] += similarity_preferred_to_item * similarity_difference * self.decay_rate
                # no correction needed
                else:
                    pass
            
            lowest_negative_number = abs(min(self.user_preferences))
    
            for i in range(len(self.user_preferences)):
                self.user_preferences[i] += lowest_negative_number
        
        # preference update
        elif modality == "P":
            option1 = self.generate_random_food()
            option2 = self.generate_random_food()
            while option1 == option2:
                option2 = self.generate_random_food()
            print("option1: ", option1.itemName)
            print("option2: ", option2.itemName)

            option1_weights = np.array([
                option1.ifSolid, option1.softness, option1.price, option1.salt_content, option1.healthiness_index, option1.sugar_content, option1.protein, option1.calories
            ])
            option2_weights = np.array([
                option2.ifSolid, option2.softness, option2.price, option2.salt_content, option2.healthiness_index, option2.sugar_content, option2.protein, option2.calories
            ])
            # choose a random number between 1 and 2
            choice = np.random.choice([1, 2])
            print("choice: ", option1.itemName if choice == 1 else option2.itemName)
            if choice == 1:
                similarities = cosine_similarity([option1_weights], [[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
            else:
                similarities = cosine_similarity([option2_weights], [[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
            self.user_preferences += self.decay_rate * similarities[0]
        
        # binary update
        elif modality == "B":
            option = self.generate_random_food()
            print("option: ", option.itemName)
            option_weights = np.array([
                option.ifSolid, option.softness, option.price, option.salt_content, option.healthiness_index, option.sugar_content, option.protein, option.calories
            ])
            choice = np.random.choice([0, 1])
            print("liked" if choice else "disliked")
            if choice:
                similarities = cosine_similarity([option_weights], [[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
                self.user_preferences += self.decay_rate * similarities[0]
            else:
                similarities = cosine_similarity([option_weights], [[item.ifSolid, item.softness, item.price, item.salt_content, item.healthiness_index, item.sugar_content, item.protein, item.calories] for item in self.food_items])
                self.user_preferences -= self.decay_rate * similarities[0]
                lowest_negative_number = abs(min(self.user_preferences))
                for i in range(len(self.user_preferences)):
                    self.user_preferences[i] += lowest_negative_number

        self.user_preferences /= np.sum(self.user_preferences)
        self.decay_rate *= self.decay_rate
    
        
        
        

        # TESTING PURPOSES

        display = [[round(self.user_preferences[x], 5), self.food_items[x].itemName] for x in range(len(self.user_preferences))]
        display.sort(reverse=True)
        print("after normalization: ", display)
        print("-----------------------------------")
    
    def update_preferences_by_gmm(self, feedback):
        weights = np.array([
            feedback.ifSolid, feedback.softness, feedback.price, feedback.salt_content, feedback.healthiness_index, feedback.sugar_content, feedback.protein, feedback.calories
        ]).reshape(1, -1)
        # print("weights: ", weights)
        likelihoods = self.gmm.score_samples(weights)
        # print("likelihoods: ", likelihoods, "log_likelihoods:" np.exp(likelihoods)) #(self.decay_rate ** len(self.user_preferences)) * likelihoods)
        # print("likelihoods: ", likelihoods)
        self.user_preferences += self.decay_rate * likelihoods
        # print("self.user_preferences: ", self.user_preferences)
        self.user_preferences /= np.sum(self.user_preferences)
        self.decay_rate *= self.decay_rate

        # TESTING PURPOSES

        # display = [[round(self.user_preferences[x], 5), self.food_items[x].itemName] for x in range(len(self.user_preferences))]
        # display.sort(reverse=True)
        # print("after normalization: ", display)
        # print("-----------------------------------")

    
test = VendingMachine()
# random_food = test.generate_random_food().itemName
# print("random_food: ", random_food)
# food_it = test.provide_demonstration(random_food)
apple = "Apple"
banana = "Banana"
pear = "Pear"
orange = "Orange"
coffee = "Coffee"

apple_it = test.provide_demonstration(apple)
banana_it = test.provide_demonstration(banana)
pear_it = test.provide_demonstration(pear)
orange_it = test.provide_demonstration(orange)
coffee_it = test.provide_demonstration(coffee)
water_it = test.provide_demonstration("Water")

# test.update_preferences_by_cosine_similarity(coffee_it, "C", apple_it)
# test.update_preferences_by_cosine_similarity(orange_it, "C", apple_it)
# test.update_preferences_by_cosine_similarity(orange_it, "C", water_it)
# test.update_preferences_by_cosine_similarity(water_it, "C", apple_it)
# test.update_preferences_by_cosine_similarity(apple_it, "C", pear_it)

for i in range(5):
    # test.update_preferences_by_cosine_similarity(None, "P", None)
    test.update_preferences_by_cosine_similarity(None, "B", None)



# test.update_preferences_by_cosine_similarity(apple_it, "C", pear_it)


# test.train_gmm()

# update_apple = test.update_preferences_by_gmm(apple_it)
# update_banana = test.update_preferences_by_gmm(banana_it)
# update_pear = test.update_preferences_by_gmm(pear_it)
# update_orange = test.update_preferences_by_gmm(orange_it)
# update_coffee = test.update_preferences_by_gmm(coffee)
# update_coffee = test.update_preferences_by_gmm(coffee)
