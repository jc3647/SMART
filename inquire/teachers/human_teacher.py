from inquire.teachers.teacher import Teacher
from inquire.utils.datatypes import Choice, Query, Feedback, Modality, Trajectory
from typing import Union
from inquire.environments.environment import CachedTask, Task
from inquire.environments.vm import FoodItem
import numpy as np

class HumanTeacher(Teacher):
    @property
    def alpha(self):
        return self._alpha
    
    def __init__(self, N, vending_machine, display_interactions: bool = False) -> None:
        super().__init__()
        self._alpha = 0.75
        self._N = N
        self._display_interactions = display_interactions
        self.vending_machine_items = vending_machine.food_items

        print("these are the items in the vending machine: ", self.vending_machine_items)
    
    def query_response(self, q: Query, task: Union[Task, CachedTask], verbose: bool=False) -> Choice:
        if q.query_type is Modality.DEMONSTRATION:
            f = self.demonstration(q, task)
        elif q.query_type is Modality.PREFERENCE:
            f = self.preference(q, task)
        elif q.query_type is Modality.CORRECTION:
            f = self.correction(q, task)
        elif q.query_type is Modality.BINARY:
            f = self.binary_feedback(q, task)
        return f
        
    def demonstration(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        print("Please provide your favorite food.")
        favorite_food = input()
        try:
            demo = self.vending_machine_items[favorite_food].get_features()
            print("these are the features of: ", favorite_food, " ", demo, ", normalized!")
            traj = Trajectory(states=np.expand_dims(demo, axis=0), actions=None, phi=demo)
            print("You have chosen: ", favorite_food)
        except:
            print("I'm sorry, I don't have that item in my vending machine.")
            return

        vending_machine_trajectories = [Trajectory(
            states=np.expand_dims(food.get_features(), axis=0),
            actions=None,
            phi=food.get_features()
        ) for food in self.vending_machine_items.values()]

        f = Feedback(
            Modality.DEMONSTRATION,
            query,
            Choice(traj, vending_machine_trajectories)
        )
        return f
    
    def preference(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        r = [qi for qi in query.trajectories]
        items = []
        for traj in r:
            closest = float("inf")
            closest_item = None
            # print("phi: ", traj.phi, "length: ", np.linalg.norm(traj.phi))
            for item in self.vending_machine_items.values():
                dist = np.linalg.norm(traj.phi - item.get_features())
                # print("dist: ", dist, "item: ", item.itemName)
                if dist < closest:
                    closest = dist
                    closest_item = item.itemName
            items.append(closest_item)
            print("-----------------")               

        # r = [Trajectory(states=self.vending_machine_items[items[0]].get_features(), actions=None, phi=self.vending_machine_items[items[0]].get_features()), 
        #      Trajectory(states=self.vending_machine_items[items[1]].get_features(), actions=None, phi=self.vending_machine_items[items[1]].get_features())]  

        r.append(Trajectory(states=self.vending_machine_items["Water"].get_features(), actions=None, phi=self.vending_machine_items["Water"].get_features()))
        print("These are your choices: ", items, "Water")#, random_items)

        fav = input("Please provide your preference: ")
        if fav == "0":
            print("You've chosen: ", items[0])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[0], r)
            )
        elif fav == "1":
            print("You've chosen: ", items[1])
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[1], r)
            )
        elif fav == "2":
            print("You've chosen: ", "Water")
            return Feedback(
                Modality.PREFERENCE,
                query,
                Choice(r[2], r)
            )
        else:
            print("I'm sorry, I didn't understand that.")
    
    # TODO - query for a food, but ask for a correction that should be another food
    def correction(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:
        traj_query = query.trajectories[0]

        closest = float("inf")
        closest_item = None
        for item in self.vending_machine_items.values():
            dist = np.linalg.norm(traj_query.phi - item.get_features())
            if dist < closest:
                closest = dist
                closest_item = item
        print("closest_item: ", closest_item.itemName, " ", closest_item.get_features())

        corrected_food = input("Please provide the correct food: ")
        try:
            corrected = self.vending_machine_items[corrected_food].get_features()
            traj = Trajectory(states=corrected, actions=None, phi=corrected)
            print("You have chosen: ", corrected_food)
        except:
            print("I'm sorry, I don't have that item in my vending machine.")
            return

        return Feedback(
            Modality.CORRECTION,
            query,
            Choice(selection=traj, options=[traj, traj_query])
        )

    def binary_feedback(self, query: Query, task: Union[Task, CachedTask]) -> Feedback:

        traj = query.trajectories[0]

        print("traj: ", traj.phi)

        closest = float("inf")
        closest_item = None
        for item in self.vending_machine_items.values():
            dist = np.linalg.norm(traj.phi - item.get_features())
            if dist < closest:
                closest = dist
                closest_item = item
        print("closest_item: ", closest_item.itemName, " ", closest_item.get_features())
        answer = input("Do you like this food? (yes/no): ")

        if answer == "yes":
            print("Nice! I'm glad you like it!")

            print("query traj: ", query.trajectories[0].phi, query.trajectories[0].states, query.trajectories[0].actions)
        
            return Feedback(
                Modality.BINARY,
                query,
                Choice(np.bool_(True), [query.trajectories[0]])
            )

            # return Feedback(
            #     query.query_type,
            #     query,
            #     Choice(np.bool_(True), [Trajectory(states=closest_item.get_features(), actions=None, phi=closest_item.get_features())])
            # )
        else:
            print("I'm sorry you didn't like it. I'll try to recommend something else.")

        
            return Feedback(
                Modality.BINARY,
                query,
                Choice(np.bool_(False), [query.trajectories[0]])
            )

            # return Feedback(
            #     Modality.BINARY,
            #     query,
            #     Choice(np.bool_(False), [Trajectory(states=closest_item.get_features(), actions=None, phi=closest_item.get_features())])
            # )