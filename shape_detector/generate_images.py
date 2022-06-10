import cv2
import numpy as np
import random
import abc
import os
import yaml


class Shape:
    def __init__(self, color):
        self.color = color
        self.image = None

    @abc.abstractmethod
    def generate_image(self):
        return

    def compute_area(self):
        return np.sum(np.sum(self.image, axis=2) != 0) / (96*96)


class Ellipse(Shape):

    def __init__(self, color):
        super().__init__(color)

    def generate_image(self):
        self.image = np.zeros((96, 96, 3), np.uint8)
        w, h = random.randint(0, 95), random.randint(0, 95)
        minim = min(w, h)
        while minim < 10:  # avoid ellipse that looks like a line
            w, h = random.randint(0, 95), random.randint(0, 95)
            minim = min(w, h)
        maxim = max(w, h)
        xc, yc = random.randint(maxim // 2, 95 - maxim // 2), random.randint(maxim // 2, 95 - maxim // 2)
        angle = random.randint(0, 360)
        cv2.ellipse(self.image, (xc, yc), (w // 2, h // 2), angle, 0, 360, self.color, -1)
        return self.image
    

class Rectangle(Shape):
    def generate_image(self):
        self.image = np.zeros((96, 96, 3), np.uint8)
        x1, y1 = random.randint(0, 95), random.randint(0, 95)
        x2, y2 = random.randint(0, 95), random.randint(0, 95)
        while x1 == x2:
            x1 = random.randint(0, 95)
        while y1 == y2:
            y1 = random.randint(0, 95)
        cv2.rectangle(self.image, (x1, y1), (x2, y2), self.color, -1)
        return self.image


class Triangle(Shape):
    def generate_image(self):
        self.image = np.zeros((96, 96, 3), np.uint8)
        x1, y1 = (random.randint(0, 95), random.randint(0, 95))
        x2, y2 = (random.randint(0, 95), random.randint(0, 95))
        x3, y3 = (random.randint(0, 95), random.randint(0, 95))
        collinear_matrix = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
        while np.linalg.det(collinear_matrix) == 0:
            x1, y1 = (random.randint(0, 95), random.randint(0, 95))
        pts = [(x1, y1), (x2, y2), (x3, y3)]
        cv2.fillPoly(self.image, np.array([pts]), self.color)
        return self.image


def generate(directory: str = 'dataset', n: int = 3006, train: bool = True):
    shapes = [Ellipse((255, 0, 0)), Ellipse((0, 255, 0)), Ellipse((0, 0, 255)),
              Rectangle((255, 0, 0)), Rectangle((0, 255, 0)), Rectangle((0, 0, 255)),
              Triangle((255, 0, 0)), Triangle((0, 255, 0)), Triangle((0, 0, 255))]
    colors = {(255, 0, 0): 'blue', (0, 255, 0): 'green', (0, 0, 255): 'red'}
    if not os.path.exists(directory):
        os.makedirs(directory)
        index = 0
        i = 1
        shape = shapes[index]
        labels = {}
        while i <= n:  # 9 * 334 = 3006
            img = shape.generate_image()
            cv2.imwrite(os.path.join(directory, f'img_{i}.jpg'), img)
            if train:
                labels[f'img_{i}'] = [type(shape).__name__, colors[shape.color], float(shape.compute_area())]
            if i % (n//9) == 0 and i != n:
                index += 1
                shape = shapes[index]
            i += 1
        if train:
            with open('labels.yaml', 'w') as file:
                yaml.dump(labels, file, sort_keys=False)

if __name__ == '__main__':
    generate()
