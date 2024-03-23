import numpy as np
from inquire.environments.environment import Environment
from inquire.utils.datatypes import Trajectory, Range
import math
from sklearn import preprocessing


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

    def get_features(self):
        return np.array([self.ifSolid, self.softness, self.price, self.salt_content, self.healthiness_index, self.sugar_content, self.protein, self.calories])
    
    def normalize_data(self, min_values, max_values):
        # print("before transformations: ", self.get_features())
        self.softness = (self.softness - (max_values['softness'] + min_values['softness']) / 2) / ((max_values['softness'] - min_values['softness']) / 2)
        self.price = (self.price - (max_values['price'] + min_values['price']) / 2) / ((max_values['price'] - min_values['price']) / 2)
        self.salt_content = (self.salt_content - (max_values['salt_content'] + min_values['salt_content']) / 2) / ((max_values['salt_content'] - min_values['salt_content']) / 2)
        self.healthiness_index = (self.healthiness_index - (max_values['healthiness_index'] + min_values['healthiness_index']) / 2) / ((max_values['healthiness_index'] - min_values['healthiness_index']) / 2)
        self.sugar_content = (self.sugar_content - (max_values['sugar_content'] + min_values['sugar_content']) / 2) / ((max_values['sugar_content'] - min_values['sugar_content']) / 2)
        self.protein = (self.protein - (max_values['protein'] + min_values['protein']) / 2) / ((max_values['protein'] - min_values['protein']) / 2)
        self.calories = (self.calories - (max_values['calories'] + min_values['calories']) / 2) / ((max_values['calories'] - min_values['calories']) / 2)
        norm = np.linalg.norm(self.get_features())  # Calculate the Euclidean length of the feature vector
        # print("standardized data: ", self.get_features())
        # print("length: ", math.sqrt(sum([x**2 for x in self.get_features()])),)
        
        # for feat in self.get_features():
        #     feat /= norm
        self.ifSolid /= norm
        self.softness /= norm
        self.price /= norm
        self.salt_content /= norm
        self.healthiness_index /= norm
        self.sugar_content /= norm
        self.protein /= norm    
        self.calories /= norm

        # print("After Euclidean length normalization: ", self.get_features())
        # print("length: ", math.sqrt(sum([x**2 for x in self.get_features()])))
        # print("itemName: ", self.itemName)

        # features = self.get_features()
        # norm = np.linalg.norm(features)  # Calculate the Euclidean length of the feature vector
        # self.ifSolid /= norm
        # self.softness /= norm
        # self.price /= norm
        # self.salt_content /= norm
        # self.healthiness_index /= norm
        # self.sugar_content /= norm
        # self.protein /= norm
        # self.calories /= norm


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

class VendingMachine(Environment):

    def __init__(self, seed, w_dim, food_items=food_items):
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self.feat_dim = w_dim
        self.action_rang = [[
            food.ifSolid,
            food.softness,
            food.price,
            food.salt_content,
            food.healthiness_index,
            food.sugar_content,
            food.protein,
            food.calories
        ] for food in food_items]
        self.action_rang = np.asarray(self.action_rang)
        self.action_rang = preprocessing.scale(self.action_rang, axis=0)
        # repopulate food_items with normalized data
        for i, food in enumerate(food_items):
            food_items[i] = FoodItem(
                food.itemName,
                self.action_rang[i][0],
                self.action_rang[i][1],
                self.action_rang[i][2],
                self.action_rang[i][3],
                self.action_rang[i][4],
                self.action_rang[i][5],
                self.action_rang[i][6],
                self.action_rang[i][7],
            )

        # self.action_range = preprocessing.scale(self.action_rang, axis=1)
        # self.action_rang = preprocessing.normalize(self.action_rang)

        # print("normalized_arr: ", self.action_rang)
        # print("self.action_rang: ", self.action_rang)

        self.state_rang = self.action_rang
        self.trajectory_length = 1
        self.food_items = {}

        # normalize food items feature weights
        # self.min_values = {
        #     "softness": float('inf'),
        #     "price": float('inf'),
        #     "salt_content": float('inf'),
        #     "healthiness_index": float('inf'),
        #     "sugar_content": float('inf'),
        #     "protein": float('inf'),
        #     "calories": float('inf')
        # }
        # self.max_values = {
        #     "softness": float('-inf'),
        #     "price": float('-inf'),
        #     "salt_content": float('-inf'),
        #     "healthiness_index": float('-inf'),
        #     "sugar_content": float('-inf'),
        #     "protein": float('-inf'),
        #     "calories": float('-inf')
        # }
        # for food in food_items:
        #     for key in self.min_values.keys():
        #         self.min_values[key] = min(self.min_values[key], getattr(food, key))
        #         self.max_values[key] = max(self.max_values[key], getattr(food, key))

        # for food in food_items:
        #     food.normalize_data(self.min_values, self.max_values)
        #     # print("normalized data: ", food.get_features(), "length: ", math.sqrt(sum([x**2 for x in food.get_features()])), "itemName: ", food.itemName)

        self.food_items = {
            food.itemName: food for food in food_items
        }

    def w_dim(self):
        return self.feat_dim

    def action_space(self):
        return self.action_rang # return the range of possible actions, which should be all items

    def state_space(self):
        return self.state_rang # this should represent the modality

    def generate_random_state(self, random_state):
        # generate a random 8D vector that has a length of 1
        # random_state = self._rng.normal(0, 1, size=(self.feat_dim,))
        # random_state = random_state / np.linalg.norm(random_state)
        # return random_state

        # print(np.zeros(self.feat_dim))
        return np.zeros(self.feat_dim) # generate a random item within the vending machine and its position

    def generate_random_reward(self, random_state):
        generated = self._rng.normal(0, 1, size=(self.feat_dim,))
        generated = generated / np.linalg.norm(generated)
        testing = math.sqrt(sum([x**2 for x in generated]))
        print("generated: ", generated, "length: ", testing)
        return generated
    
    # def generate_random_reward(self, random_state):
    #     generated = np.abs(self._rng.normal(0, 1, size=(self.feat_dim,)))
    #     generated /= np.max(generated)  # Normalize to ensure values are between 0 and 1
    #     testing = math.sqrt(sum([x**2 for x in generated]))
    #     print("generated: ", generated, "length: ", testing)
    #     return generated

    def optimal_trajectory_from_w(self, start_state, w):
        # print("expand_dims: ", np.expand_dims(w, axis=0))
        return Trajectory(states=np.expand_dims(w, axis=0), actions=None, phi=w)

    def trajectory_rollout(self, start_state, actions):
        action = actions[-self.feat_dim:]

        state = action / np.linalg.norm(action)

        return Trajectory(states=np.expand_dims(state, axis=0), actions=None, phi=state)

    def features_from_trajectory(self, trajectory):
        return trajectory.states[-1]

    def distance_between_trajectories(self, a, b):
        cos = np.dot(a.phi, b.phi) / (np.linalg.norm(a.phi) * np.linalg.norm(b.phi))
        return np.arccos(np.clip(cos, -1, 1)) / math.pi

    def visualize_trajectory(self, start_state, trajectory):
        print(trajectory.states[-1])