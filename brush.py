import cv2
from network import CharacterRecognizer
import numpy as np
import torch
import tkinter as tk
import matplotlib.pyplot as plt
import math
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

BRUSH_RADIUS = 2
BRUSH_WEIGHT = 1
IMG_SIZE = 28
PIXEL_SIZE = 15

PATH = './model/character_recognizer.pth'

LABELS = {
  0: "zero",
  1: "one",
  2: "two",
  3: "three",
  4: "four",
  5: "five",
  6: "six",
  7: "seven",
  8: "eight",
  9: "nine",
  10: "capital A",
  11: "capital B",
  12: "captial C",
  13: "captical D",
  14: "captical E",
  15: "captical F",
  16: "captical G",
  17: "captical H",
  18: "captical I",
  19: "captical J",
  20: "captical K",
  21: "captical L",
  22: "captical M",
  23: "captical N",
  24: "captical O",
  25: "captical P",
  26: "captical Q",
  27: "captical R",
  28: "captical S",
  29: "captical T",
  30: "captical U",
  31: "captical V",
  32: "captical W",
  33: "captical X",
  34: "captical Y",
  35: "captical Z",
  36: "lower case a",
  37: "lower case b",
  38: "lower case d",
  39: "lower case e",
  40: "lower case f",
  41: "lower case g",
  42: "lower case h",
  43: "lower case n",
  44: "lower case q",
  45: "lower case r",
  46: "lower case t",
}

class Brush:

  def __init__(self, master):
    
    self.master = master
    self.image = [[0] * IMG_SIZE for _ in range(IMG_SIZE)]

    self.canvas = tk.Canvas(master, width=IMG_SIZE*PIXEL_SIZE, height=IMG_SIZE*PIXEL_SIZE, bg='black')
    self.canvas.pack()
    self.canvas.bind('<Button-1>', self.start_stroke)
    self.canvas.bind('<B1-Motion>', self.draw)

    self.clear_button = tk.Button(master, text='clear canvas', command=self.clear_canvas)
    self.clear_button.pack()

    self.save_button = tk.Button(master, text='make prediction', command=self.make_prediction)
    self.save_button.pack()

    self.prediction_label = tk.Label(master, text='prediction:')
    self.prediction_label.pack()

    self.prediction = tk.Text(master, height=5, width=50)
    self.prediction.pack()

    self.model = CharacterRecognizer()
    self.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    self.model.eval()

    self.prev_x = None
    self.prev_y = None
    self.new_stroke = True

  def start_stroke(self, event):
    self.prev_x = event.x // PIXEL_SIZE
    self.prev_y = event.y // PIXEL_SIZE
    self.new_stroke = True

  def draw(self, event):
    x = event.x // PIXEL_SIZE
    y = event.y // PIXEL_SIZE

    if self.new_stroke:
      self.new_stroke = False
    else:
      # interpolate between previous and current mouse positions
      steps = max(abs(x - self.prev_x), abs(y - self.prev_y)) + 1
      for i in range(steps):
        interp_x = self.prev_x + (x - self.prev_x) * i / steps
        interp_y = self.prev_y + (y - self.prev_y) * i / steps
        self.draw_pixel(interp_x, interp_y)

    self.prev_x = x
    self.prev_y = y
    

  def draw_pixel(self, x, y):

    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        dist = (i - x) ** 2 + (j - y) ** 2

        if dist < 1:
          dist = 1

        dist = dist ** 3.5

        self.image[i][j] += (BRUSH_RADIUS / dist) * BRUSH_WEIGHT

        if self.image[i][j] > 1:
          self.image[i][j] = 1

        if self.image[i][j] < 0.001:
          self.image[i][j] = 0

    self.update_canvas()

  def update_canvas(self):
    self.canvas.delete('all') 
    # go through every single element in the image and create a rectangle. brightness depends on value
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        color = int(self.image[i][j] * 255)
        self.canvas.create_rectangle(i * PIXEL_SIZE, j * PIXEL_SIZE, (i + 1) * PIXEL_SIZE, (j + 1) * PIXEL_SIZE, fill=f'#{color:02x}{color:02x}{color:02x}', outline=f'#{color:02x}{color:02x}{color:02x}')

  def clear_canvas(self):
    
    self.image = [[0] * IMG_SIZE for _ in range(IMG_SIZE)]
    self.canvas.delete('all')

    self.prediction.delete(1.0, tk.END)
   
    self.prev_x = None
    self.prev_y = None
    self.new_stroke = True

  def make_prediction(self):

    self.preprocess()
    self.update_canvas()
    
    image_numpy = np.array(self.image)
    image_numpy = np.transpose(image_numpy)
    

    image_tensor = torch.from_numpy(image_numpy)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).float()

    output = self.model(image_tensor)


    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    self.prediction.delete(1.0, tk.END)
    self.prediction.insert(tk.END, LABELS[predicted_class])

    self.grad_cam(image_tensor=image_tensor)
    
  # look at documents 
  def grad_cam(self, image_tensor):
    
    cam = GradCAM(model=self.model, target_layers=self.model.conv3)
    grayscale_cam = cam(input_tensor=image_tensor)[0] # cam returns a tensor which was really fucking shit up half the day :)
  
    image = image_tensor.squeeze().numpy()  
    image = image.astype(np.float32)
    image = np.stack([image] * 3, axis=-1)
    
    heatmap = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    plt.imshow(heatmap)
    plt.show()
    
  # the point is so center the "image" because the model isnt used to characters being offset.
  def preprocess(self):
    
    min_x = 0
    max_x = 0
    
    min_y = 0
    max_y = 0
    
    # edge case if the first element for the min.
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        
        if self.image[i][j] > 0:
          min_x = j
          min_y = i
          break
          
  
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        
        if self.image[i][j] > 0:
          min_x = min(min_x, j)
          min_y = min(min_y, i)
          
          max_x = max(max_x, j)
          max_y = max(max_y, i)
          
    image_x_center = (min_x + max_x) // 2
    image_y_center = (min_y + max_y) // 2
    
    offset_x = (IMG_SIZE // 2) - image_x_center - 1
    offset_y = (IMG_SIZE // 2) - image_y_center - 1

    centered_image = [[0] * IMG_SIZE for _ in range(IMG_SIZE)]
    
    for i in range(IMG_SIZE):
      for j in range(IMG_SIZE):
        
        if self.image[i][j] > 0:
          
          centered_image[j + offset_x][i + offset_y] = self.image[i][j]
    
    self.image = np.transpose(np.array(centered_image)).tolist()

      
      
            



