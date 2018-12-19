#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import datetime
import json
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from pysmt.shortcuts import Symbol, And, Not, is_sat, GE, Int, TRUE, Equals, Or, FALSE, is_valid, Times, LE, Bool
from pysmt.constants import (Fraction, is_python_integer, is_python_boolean)
from pysmt.typing import INT
#from z3 import *

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training #ANITHA CHANGED THIS FOR DEMO
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
#paths, action_list
path_list = []
actionlist = [1,0]
img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames
epsilon = 2 # this is just done to offset the checking around the brid
	
def buildmodel():	
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    print("---------TRAINING THE MODEL--------")
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    #safeRL_player: the state of the player/bird is got from the first frame for safe RL
    x_t, r_0, terminal, safeRL_player = game_state.frame_step(do_nothing)
	#SafeRL: The game's first state -- the image -- is copied here for accessing Safety of next action
    safeRL_game_image = x_t
        
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        #print (epsilon)
        if t % FRAME_PER_ACTION == 0:
            #if random.random() <= epsilon:
            if 2 <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
                q = a_t
            else: #Q learning
                #print ("***********in q learning************* *****")
                q = model.predict(s_t)      #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q       #maximixing Q learning
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #SafeRL: -------------------------------------------------------------------start----------			
		# offsetting the x position of the birl to take its width into consideration
        bird_x_pos = safeRL_player['x'] + 30 
		#the image is transposed since X and Y are interchanged in the original image. 
        safeRL_game_image = np.einsum("kli->lki", safeRL_game_image)

		# we are making a dynamic object with all the items we want from the game
        dynamical_parameters = {'step':t, 'pos_x':bird_x_pos, 'pos_y':safeRL_player['y'],
		'vel_x':game_state.pipeVelX,'vel_y':game_state.playerVelY,
		'acc_y_up':game_state.playerFlapAcc,'acc_y_down':game_state.playerAccY,
		'max_vel_y':game_state.playerMaxVelY, 'min_vel_y':game_state.playerMinVelY}

        bird_param = game_state.birdimage_param
		# safest action is picked using the current model, action list and predicted rewards (q)
        a_t = safe_action_picker(safeRL_game_image, dynamical_parameters, q, bird_param)

        # the next step of the game is played with the selected action
        x_t1_colored, r_t, terminal, safeRL_player = game_state.frame_step(a_t)
		#the game state -- the image -- is copied here for accessing Safety of next action
        safeRL_game_image = x_t1_colored
        #SafeRL: -------------------------------------------------------------------end----------			

		
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            #print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 100 == 0:
            print("The model is saved at t = ", t)
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
			
        #print("TIMESTEP", t, "/ STATE", state, \
        #    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #    "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")
			 
def safe_action_picker(game_image, dynamical_parameters, predicted_reward, bird_param):
    #print("")
    #print("-------------------------------------------------------------------------STEP : ", dynamical_parameters['step'])
    #print("")
    #sort the actions per predicted_reward
    sorted_actions = np.argsort(predicted_reward)[0]
    #print("sorted_actions   ", sorted_actions)
    smt_result = 0    
    player_pos_x = dynamical_parameters['pos_x']
    player_pos_y = dynamical_parameters['pos_y']	
    flapup_paths = None
    godown_paths = None
		
    if (dynamical_parameters['step'] < 3):
         sorted_actions[1] = 1
         sorted_actions[0] = 0
         #print(" FIRST FEW STEPS SAFELY FLAP UP   ")
         return sorted_actions
    if (dynamical_parameters['step'] < 10):
         sorted_actions[1] = 1
         sorted_actions[0] = 0
         #print(" FIRST FEW STEPS SAFELY FLAP UP   ")
         #return sorted_actions
		 
		 

    #print(" Before Mask Bird   ", datetime.datetime.now().strftime("%H:%M:%S"))
	#masking the bird in the image before processing 
    masked_image = copy.deepcopy(game_image)
    bird_param ['x'] = dynamical_parameters['pos_x']
    bird_param ['y'] = dynamical_parameters['pos_y']
    masked_image = mask_bird (masked_image, bird_param)
    #print(" After Mask Bird   ", datetime.datetime.now().strftime("%H:%M:%S"))
    
	
# default I am just setting is 
    for act in sorted_actions:
        #print(" Before SMT   ", datetime.datetime.now().strftime("%H:%M:%S"))
        smt_result, smt_flapup_paths, smt_godown_paths = safeRL_callSMT(act, player_pos_x, player_pos_y, dynamical_parameters, masked_image, bird_param)
        #print(" After SMT   ", datetime.datetime.now().strftime("%H:%M:%S"))
        exists_path = 0        
        if (act == 1):
             flapup_paths = smt_flapup_paths		
        if (act == 0):
             godown_paths = smt_godown_paths		
		
        #Smt method returns 1 for flap up safe, 2 for do nothing and 0 for no action safe
        if (smt_result == 1):
             sorted_actions[1] = 1
             sorted_actions[0] = 0
             #print("  ::::::::::::::::::::::::  SAFE ACTION ^^^^^^^^^^^^^^^^^^ 1  FLAP UP   ")
             exists_path = 1
             break
			 
        if (smt_result == 2) :
             sorted_actions[0] = 1
             sorted_actions[1] = 0
             #print("  ::::::::::::::::::::::::  SAFE ACTION VVVVVVVVVVVVVVVVVV 0   do nothing ")
             exists_path = 1
             break
    #print(" exists_path :::   ", exists_path)			 
    if (exists_path == 0):
        if (player_pos_y <= 200):
             sorted_actions[0] = 1
             sorted_actions[1] = 0
             #print("  ::::::::::::::::::::::::  DEFA ACTION :^^^^^^^^^^^^^^^^^^ 0  do nothing based on bird position ")		
        if (player_pos_y > 200):
             sorted_actions[1] = 1
             sorted_actions[0] = 0
             #print("  ::::::::::::::::::::::::  DEFA ACTION :VVVVVVVVVVVVVVVVVV 1  Flap up  based on bird position ")
    
    #print(" Before plot image   ", datetime.datetime.now().strftime("%H:%M:%S"))
    visual_display = game_image
    if (flapup_paths is not None):
         direction = 1
         visual_display = plot_points(visual_display, flapup_paths, direction, bird_param ['w'], bird_param ['h'])

    if (godown_paths is not None):    
         direction = 2
         visual_display = plot_points(visual_display, godown_paths, direction,  bird_param ['w'], bird_param ['h'])		 		
    
    #if (dynamical_parameters['step'] > 30):    
    show_image(visual_display, bird_param ['x'] , bird_param ['y'],bird_param ['w'] , bird_param ['h'] )
	#print(" After plot image   ", datetime.datetime.now().strftime("%H:%M:%S"))
    return sorted_actions
		
#recusrive code by Sina	
def recursive_path_generator (pos_x, pos_y, vel_y, action_ctr, dynamical_parameters, path_list, path_depth, path = None):
 
    pos_x_st = pos_x
    pos_y_st = pos_y
    vel_y_st = vel_y
    dynamical_parameters_st = dict(dynamical_parameters)
    path_st = list(path)
    action_ctr += 1
    for act in actionlist:
         pos_x, pos_y, vel_y = predict_next_steps(pos_x_st, pos_y_st, vel_y_st, act, dynamical_parameters_st)
         #path = path[:action_ctr]
         path_st = list(path)
         path_st.append((pos_x, pos_y))
         #print("  len(path_list)  ", len(path_list), "  path_list : ", path_list)
         #if (len(path_list) > (path_depth*4)):
         #    return path_list
		 
         if (len(path_st) < path_depth):
             path_list = recursive_path_generator (pos_x, pos_y, vel_y, action_ctr, dynamical_parameters, path_list, path_depth, path_st)
         else:
             path_list.append(path_st)

             
    
    return path_list
	
#recusrive code by Sina
def predict_next_steps (pos_x, pos_y, vel_y, action, dynamical_parameters):
    # compute birds next x (constant Velocity)-------------------------------\/
    x_new = pos_x - dynamical_parameters['vel_x'] # pipe is coming towards the bird
    # compute birds velocity (Contstant Acceleration)------------------------\/
    current_playerVelY = vel_y
    if (action == 0): # goes down 		
         vel_y += dynamical_parameters['acc_y_down']
         vel_y = min(vel_y, dynamical_parameters['max_vel_y'])
         # compute birdes y using averaged velocity (Contstant Acceleration)------\/
         #y_new = pos_y + int((vel_y + current_playerVelY)/2)
         vel_computation = int((vel_y + current_playerVelY)/2) #ANITHA: I need to write an explanation about this logic. 
         if (vel_computation > 0):
             y_new = pos_y + vel_computation
         else:
             y_new = pos_y - vel_computation
    else:	
         vel_y += dynamical_parameters['acc_y_up']
         vel_y = max(vel_y, dynamical_parameters['min_vel_y'])
         # compute birdes y using averaged velocity (Contstant Acceleration)------\/
         vel_computation = int((vel_y + current_playerVelY)/2)#ANITHA: I need to write an explanation about this logic. 
         if (vel_computation > 0):
             y_new = pos_y - vel_computation
         else:
             y_new = pos_y + vel_computation
			 
    #y_new = pos_y + int((vel_y + current_playerVelY)/2)
    return x_new, y_new, vel_y
    
def safeRL_callSMT(action, start_pos_x, start_pos_y, dynamical_parameters, image, bird_param):

    flapup_paths = None
    godown_paths = None
    smt_result = 0
    path_depth = 8
    if (action == 1):
         #==========================================================================================================
         # properties for safe flap up action
         #----------------------------------------------------------------------------------------------------------
         #property 1: safe_upper_height is the safe max altitude of the bird, depending upon the game
         safe_max_height = 70
         safety_prop_1 = GE(Int(dynamical_parameters['pos_y']), Int(safe_max_height))
		 
         #property 2:if there is no escape path if it takes action flap, then do not take it.    
         # number_of_ states to check for safe movement depends upon the game speed, bird vel etc.         
         path_list = []
		 
         temp_dynamical_parameters = {}
         temp_dynamical_parameters['pos_x'], temp_dynamical_parameters['pos_y'], temp_dynamical_parameters['vel_y'] = predict_next_steps (dynamical_parameters['pos_x'],dynamical_parameters['pos_y'], dynamical_parameters['vel_y'], action, dynamical_parameters)
         path = [(temp_dynamical_parameters['pos_x'],temp_dynamical_parameters['pos_y'])]
         action_ctr = 1
         flapup_paths = recursive_path_generator (temp_dynamical_parameters['pos_x'], temp_dynamical_parameters['pos_y'], temp_dynamical_parameters['vel_y'], action_ctr, dynamical_parameters, path_list, path_depth, path=path)
         is_flap_up_path_clear = is_path_safe(image, flapup_paths, bird_param['w'],bird_param['h'], action)
         safety_prop_2 = (Bool(is_flap_up_path_clear))
         #----------------------------------------------------------------------------------------------------------	

         #property 3: Safe vertical distance of bird from pipe - when it goes through the pipe  
         # min distance between pipe and bird.         
         safe_v_pipe_dist = 15     
         bird_epsilon = 10		 
         is_upper_pipe_far = is_safe_from_upper_pipe(image, start_pos_x, start_pos_y, (bird_param['w']+bird_epsilon), (bird_param['h']+bird_epsilon), safe_v_pipe_dist)
         safety_prop_3 = (Bool(is_upper_pipe_far))
         #----------------------------------------------------------------------------------------------------------	

         #print(" Before smt  ", datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3])
         properties_for_flap_action = And(safety_prop_1, safety_prop_2, safety_prop_3)
         res = is_valid(properties_for_flap_action, solver_name="z3")
         #print("flap  :", properties_for_flap_action, "      " , res)
         if res:
             smt_result = 1
		 
    if (action == 0):
		 #----------------------------------------------------------------------------------------------------------
		 # properties for safe no action
		 #----------------------------------------------------------------------------------------------------------    
		 #property : safe_min_height is the min altitude of the bird
		 #safe_min_height value depends on the game. Its hardcoded to 500 for this game. 
         safe_min_height = 300
         safety_prop_10 = LE(Int(dynamical_parameters['pos_y']), Int(safe_min_height))
		 
         #property :if there is no escape path if it takes action no flap, then do not take it.    
         # number_of_ states to check for safe movement depends upon the game speed, bird vel etc.         
         #next_possible_actions = down_action_sequence_for_path(path_depth)
         #godown_paths = path_generator (start_pos_x, start_pos_y + 20,dynamical_parameters,  next_possible_actions)

         path_list = []
         temp_dynamical_parameters = {}
         temp_dynamical_parameters['pos_x'], temp_dynamical_parameters['pos_y'], temp_dynamical_parameters['vel_y'] = predict_next_steps (dynamical_parameters['pos_x'],dynamical_parameters['pos_y'], dynamical_parameters['vel_y'], action, dynamical_parameters)
         path = [(temp_dynamical_parameters['pos_x'],temp_dynamical_parameters['pos_y'])]
         action_ctr = 1
         godown_paths = recursive_path_generator (temp_dynamical_parameters['pos_x'], temp_dynamical_parameters['pos_y'], temp_dynamical_parameters['vel_y'], action_ctr, dynamical_parameters, path_list, path_depth, path=path)
         #print(" godown_paths  : ", godown_paths)
         is_go_down_clear = is_path_safe(image, godown_paths, bird_param['w'],bird_param['h'], action)
         safety_prop_11 = (Bool(is_go_down_clear))
         #----------------------------------------------------------------------------------------------------------

		 #property 3: Safe vertical distance of bird from pipe - when it goes through the pipe  
         # min distance between pipe and bird.         
         safe_v_pipe_dist = 15        
         bird_epsilon = 10
         is_lower_pipe_far = is_safe_from_lower_pipe(image, start_pos_x, start_pos_y, (bird_param['w']+bird_epsilon), (bird_param['h']+bird_epsilon), safe_v_pipe_dist)
         safety_prop_12 = (Bool(is_lower_pipe_far))
         #----------------------------------------------------------------------------------------------------------	
		 
         properties_for_no_action = And(safety_prop_10, safety_prop_11, safety_prop_12)
         res = is_valid(properties_for_no_action, solver_name="z3")
         #print("down  :", properties_for_no_action, "      " , res)
         if res:
             smt_result = 2

    return smt_result, flapup_paths, godown_paths	
	
def mask_bird (img, bird_params):
	for w in range(bird_params['w']):
         for h in range(bird_params['h']):	   
             img[bird_params['y']+h][bird_params['x']-w][0] = 0
             img[bird_params['y']+h][bird_params['x']-w][1] = 0
             img[bird_params['y']+h][bird_params['x']-w][2] = 0
	return img
	
def is_path_safe(img, paths, birdwidth, birdheight, direction):

    no_safe_path_count = 0
    is_unsafe = True
    this_path_safe = True
    path_count = 0
    safe_path_count = 0
    for each_path in paths:
         cnt_coordinate = 0
         for each_coordinate in each_path:
             x, y = each_coordinate
             cnt_coordinate += 1	 
             if(direction == 1):
                 f_x = x
                 f_y = y-epsilon
                 b_x = x - birdwidth
                 b_y = y - epsilon
		 
             if(direction == 0):
                 f_x = x+epsilon
                 f_y = y+int(birdheight/2)+epsilon
                 b_x = x - epsilon - birdwidth
                 b_y = y + int(birdheight/2) + epsilon	 

             is_unsafe =              (img[f_y][f_x][0] > 0 or img[b_y][b_x][0] > 0)
             is_unsafe = is_unsafe or (img[f_y][f_x][1] > 0 or img[b_y][b_x][1] > 0)			 
             is_unsafe = is_unsafe or (img[f_y][f_x][2] > 0 or img[b_y][b_x][2] > 0)			 

             if (is_unsafe):
                 no_safe_path_count += 1
                 this_path_safe = False
                 break

    if ((no_safe_path_count) >= len(paths)/2):
         #print("X X X X X PATH HAS OBSTACLES X X X X")
         return False
    else:
         #print("! ! ! ! ! PATH IS CLEAR ! ! ! ! ! ! !")
         return True
			 
    return False

def is_safe_from_upper_pipe(game_image, bird_pos_x, bird_pos_y, bird_width, bird_height, safe_pipe_dist):

    bird_pos_y = bird_pos_y - safe_pipe_dist
    if ((game_image[bird_pos_y][bird_pos_x][0] > 0 and game_image[bird_pos_y][bird_pos_x-(int(bird_width/2))][0] > 0 and game_image[bird_pos_y][bird_pos_x - bird_width][0] > 0) or    (game_image[bird_pos_y][bird_pos_x][1] > 0 and game_image[bird_pos_y][bird_pos_x-(int(bird_width/2))][1] > 0 and game_image[bird_pos_y][bird_pos_x - bird_width][1] > 0)):
        #print("X X X X X X X X  TOO CLOSE TO THE UPPER PIPE X X X X X X X X   ")   
        return False
    else:
        #print("!!!!!!!!!!!!!!! SAFE DISTANCE FROM THE UPPER PIPE !!!!!!!!!!!!!!!")
        return True	

def is_safe_from_lower_pipe(game_image, bird_pos_x, bird_pos_y, bird_width, bird_height, safe_pipe_dist):

    bird_pos_y = bird_pos_y + bird_height + safe_pipe_dist
    if ((game_image[bird_pos_y][bird_pos_x][0] > 0 and game_image[bird_pos_y][bird_pos_x - int(bird_width/2)][0] > 0 and game_image[bird_pos_y][bird_pos_x - bird_width][0] > 0) or    (game_image[bird_pos_y][bird_pos_x][1] > 0 and game_image[bird_pos_y][bird_pos_x - int(bird_width/2)][1] > 0 and game_image[bird_pos_y][bird_pos_x - bird_width][1] > 0)):
        #print("X X X X X X X X  TOO CLOSE TO THE LOWER PIPE X X X X X X X X   ")         
        return False
    else:
        #print("!!!!!!!!!!!!!!! SAFE DISTANCE FROM THE LOWER PIPE !!!!!!!!!!!!!!!")
        return True
		
def show_image(display_image, x, y, birdwidth,birdheight):
     #print("in show image n, ow")
	 
	 # plot the pupper pipe hit

     safe_pipe_dist = 15
     display_image[y - safe_pipe_dist][x][2] = 255
     display_image[y - safe_pipe_dist][x - int(birdwidth/2)][2] = 255
     display_image[y - safe_pipe_dist][x - birdwidth][2] = 255

     display_image[y + birdheight + safe_pipe_dist][x][0] = 255
     display_image[y + birdheight + safe_pipe_dist][x - int(birdwidth/2)][0] = 255
     display_image[y + birdheight + safe_pipe_dist][x - birdwidth][0] = 255 
	 
     plt.figure()
     small_image = display_image[y-100:y+150, x-50:x+200]
     plt.imshow(small_image)
     plt.show()
	
def plot_points(display_image, paths, direction, birdwidth, birdheight):
     for each_path in paths:
         for each_coordinate in each_path:
             x, y = each_coordinate
             display_image = display_cross(display_image,y, x, direction, birdwidth, birdheight)
     return display_image
	
def display_cross(image,y, x, direction, birdwidth, birdheight):
    #print("display cross ", direction)
    front_r = 255
    front_g = 255
    front_b = 255
    back_r = 255
    back_g = 255
    back_b = 255
	
    if (direction == 1):
         front_r = 255
         front_g = 0
         front_b = 0
         back_r = 255
         back_g = 200
         back_b = 130
         f_x = x
         f_y = y-epsilon
         b_x = x - birdwidth
         b_y = y - epsilon

    if (direction == 2):
         front_r = 0
         front_g = 0
         front_b = 255
         back_r = 130
         back_g = 255
         back_b = 255
         f_x = x+epsilon
         f_y = y+int(birdheight/2)+epsilon
         b_x = x - epsilon - birdwidth
         b_y = y + int(birdheight/2) + epsilon	 
		 
    image[f_y][f_x][0] = front_r
    image[f_y][f_x][1] = front_g
    image[f_y][f_x][2] = front_b

    image[f_y-1][f_x-1][0] = front_r
    image[f_y-1][f_x-1][1] = front_g
    image[f_y-1][f_x-1][2] = front_b

    image[f_y-1][f_x+1][0] = front_r
    image[f_y-1][f_x+1][1] = front_g
    image[f_y-1][f_x+1][2] = front_b

    image[f_y+1][f_x-1][0] = front_r
    image[f_y+1][f_x-1][1] = front_g
    image[f_y+1][f_x-1][2] = front_b

    image[f_y+1][f_x+1][0] = front_r
    image[f_y+1][f_x+1][1] = front_g
    image[f_y+1][f_x+1][2] = front_b
		

		
    image[b_y][b_x][0] = back_r
    image[b_y][b_x][1] = back_g
    image[b_y][b_x][2] = back_b

    image[b_y-1][b_x-1][0] = back_r
    image[b_y-1][b_x-1][1] = back_g
    image[b_y-1][b_x-1][2] = back_b

    image[b_y-1][b_x+1][0] = back_r
    image[b_y-1][b_x+1][1] = back_g
    image[b_y-1][b_x+1][2] = back_b

    image[b_y+1][b_x-1][0] = back_r
    image[b_y+1][b_x-1][1] = back_g
    image[b_y+1][b_x-1][2] = back_b

    image[b_y+1][b_x+1][0] = back_r
    image[b_y+1][b_x+1][1] = back_g
    image[b_y+1][b_x+1][2] = back_b
    #print(image[y][x][0])
    return image
	
def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)
	
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
	
