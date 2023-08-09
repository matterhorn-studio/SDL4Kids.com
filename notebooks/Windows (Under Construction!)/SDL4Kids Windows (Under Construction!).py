import serial
import time
import pandas as pd
import numpy as np

ser = serial.Serial()
ser.baudrate = 115200
# ls /dev/cu.*
#ser.port = "/dev/cu.usbmodem1102" #you will need to update this based on the serial port, you can use Mu for this
ser.port = "COM3" 
BLACK = (0, 0, 0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

import pygame

pygame.init()
font = pygame.font.SysFont(None, 24)
surface = pygame.display.set_mode((1000, 400))

# Optimisation goal
goal = np.array([70,125,50])

def setup_pygame():
    ser.open()

    #Goal
    goal_color = tuple(goal)
    pygame.draw.rect(surface, goal_color, pygame.Rect(0, 0, 100, 100))
    pygame.draw.rect(surface, BLACK, pygame.Rect(0, 98, 100, 2))
    img = font.render('Goal', True, BLACK)
    surface.blit(img, (20, 20))

    pygame.draw.rect(surface, WHITE, pygame.Rect(510, 5, 2, 95))
    pygame.draw.rect(surface, RED, pygame.Rect(510, 105, 2, 95))
    pygame.draw.rect(surface, GREEN, pygame.Rect(510, 205, 2, 95))
    pygame.draw.rect(surface, BLUE, pygame.Rect(510, 305, 2, 95))

def reset_graphs():
    pygame.draw.rect(surface, BLACK, pygame.Rect(100, 0, 100, 500))
    pygame.draw.rect(surface, BLACK, pygame.Rect(520, 0, 500, 500))

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

def calculate_loss(goal, sample):
    measured = np.array([sample["R"], sample["G"], sample["B"]])
    return - np.linalg.norm(goal - measured)
# Logging
loss_denominator = 100
def draw_strategy_info(name, run):
    img = font.render(f"Strategy {name} (Run {run})", True, WHITE, BLACK)
    surface.blit(img, (1,1))

def draw_sampleID(sample_ID):
    img = font.render(f"Sample {sample_ID}", True, WHITE, BLACK)
    surface.blit(img, (1,110))

def draw_loss(sample, i):
    surface.set_at( (100 + i, 100 + 1*int(100*int(sample['outcome'])/loss_denominator) ), WHITE)
    pygame.display.flip()

def draw_max_loss(samples, i):
    pos = (520 + i, 100 + 1*int(100*int(np.max(samples['outcome']))/loss_denominator) )
    pygame.draw.circle(surface, WHITE, pos, 1)
    img = font.render(f"{np.max(samples['outcome'])}/{255}", True, WHITE, BLACK)
    surface.blit(img, (900, 0 + 50))
    pygame.display.flip()

def draw_RGB(sample, i):
    for c, color, offset in [("R",(255,0,0), 100),("G",(0,255,0),200),("B",(0,0,255), 300)]:
        surface.set_at((520 + i,  offset + int(100 * int(sample[c]) / 255)), color)
        img = font.render(f"{int(sample[c])}/{255}", True, WHITE, BLACK)
        surface.blit(img, (900, offset + 50))
    pygame.display.flip()

# Synthesis

def set_color(data):
    color = (data["R"], data["G"], data["B"])
    print(f"(Synthesis) Set color to:{color}")
    pygame.draw.rect(surface, color, pygame.Rect(0, 100, 500, 500))
    pygame.display.flip()

# Characterisation

def measure_outcome():
    time.sleep(0.4)
    print('(Characterisation) Measure RGB sample:')
    ser.write(b",")
    # serial_data = str(ser.readline()).split(" ")[0][2:]
    serial_data = str(ser.readline().decode('utf8'))
    new_sample = {}
    while True:
        color, val = serial_data.split(":")
        #color = 'B'
        #val = 99
        new_sample[str(color)] = int(val)
        #print(new_sample)
        if color == "B":
            break
        else:
            serial_data = str(ser.readline().decode('utf8'))

    new_sample["outcome"] = calculate_loss(goal, new_sample)
    new_sample["time"] = time.time()

    print(new_sample)
    return new_sample

# Strategies

from sklearn.model_selection import ParameterGrid
param_range = np.linspace(0,255,5)
param_grid = {key:param_range for key in ["R", "G", "B"] }
grid = list(ParameterGrid(param_grid))
def grid_search(samples, sample_ID):
    return grid[sample_ID]

def random_search(samples, sample_ID):
    return {key:np.random.randint(0,255) for key in ["R", "G", "B"] }

def calculate_recommendation(samples):
    train_X = samples[["R","G","B"]]
    train_Y = samples[["outcome"]]

    # Assuming train_X and train_Y are pandas DataFrames
    train_X_np = train_X.to_numpy(dtype=np.float64)
    train_Y_np = train_Y.to_numpy(dtype=np.float64)

    train_X_tensor = torch.from_numpy(train_X_np)
    train_Y_tensor = torch.from_numpy(train_Y_np)
    gp = SingleTaskGP(train_X_tensor, train_Y_tensor)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    from botorch.acquisition import UpperConfidenceBound
    UCB = UpperConfidenceBound(gp, beta=0.1)
    from botorch.optim import optimize_acqf
    bounds = torch.stack([torch.zeros(3), torch.ones(3)*255])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    candidate = candidate[0]
    return {"R": candidate[0], "G": candidate[1], "B": candidate[2]}
def BO_search(samples, sample_ID):
    return calculate_recommendation(samples)

strategies = [BO_search, grid_search, random_search ]
# strategies = [grid_search]
strategies = [BO_search]

def a_single_run(strategy, samples, sample_ID):
    get_recommendation = strategy

    # Characterise
    new_sample = measure_outcome()
    new_sample["ID"] = sample_ID
    samples = pd.concat([samples, pd.DataFrame([new_sample])], ignore_index=True)

    # Plan
    candidate = get_recommendation(samples, sample_ID)

    # Synthesize
    set_color(candidate)

    # Log
    draw_sampleID(sample_ID)
    draw_loss(new_sample, sample_ID)
    draw_RGB(new_sample, sample_ID)
    draw_max_loss(samples, sample_ID)

    return samples

def run_strategy(s):
    samples = pd.DataFrame(columns=['R', 'G', 'B', 'outcome', 'time', 'ID'])
    set_color({key:np.random.randint(0,255) for key in ["R", "G", "B"] })
    new_sample = measure_outcome()
    new_sample["ID"] = 0
    print(new_sample)
    samples = pd.concat([samples, pd.DataFrame([new_sample])], ignore_index=True)
    # set_color({"R": 70, "G": 125, "B": 50})
    sample_ID = 0
    while True:
        sample_ID = sample_ID + 1
        if sample_ID > 120:
            return samples
        samples = a_single_run(s, samples, sample_ID)

import random
results = "results/"

if __name__ == "__main__":
    setup_pygame()

    for s in strategies:
        for i in range(1):
            draw_strategy_info(s.__name__, i)
            random.shuffle(grid)
            samples = run_strategy(s)
            # samples.to_csv(f"{results}{s.__name__}_{i}.csv")
            reset_graphs()
    pygame.time.wait(1000*30)

    ser.close()

print(samples)  #find other ways to save the results