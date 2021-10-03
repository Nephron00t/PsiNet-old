import socket
import threading
import numpy as np
import json
import pickle
# IMPORTING MY MADE UP FUNCTIONS"
from functions import stimtypes, get_psd, get_erd, get_binary_decision, client_save_data, \
    stimulation_decision, compute_ccorr, update_Wmatrix, do_Wmatrix_operations, set_timers, keep_current_stim_data, \
    update_timers, update_cooldowns, server_save_data, server_save_report, reset_vars  # THESE ARE MY MADE UP FUNCTIONS

#######################################import bluepy.btle as btle
import time
import multiprocessing
import numpy as np
import pandas as pd
import datetime
import pytz
import time
import brainflow
import numpy as np
import pandas as pd
from copy import copy
import matplotlib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from numpy.fft import fft, ifft
from astropy.stats import circcorrcoef
from enum import Enum
###########################################################################
#for alg
init = False
cooldown = [10, 10, 10]
cooldown_time = 20  ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!pls change this to 30 * 60 if not testing
saving = True

stim_threshold = 0.59
reward_value = 0.05

tbuffer = 10
MOTORt = 2*tbuffer  # 1-12 min + tbuffer
COGNITIVEt = 2*tbuffer  # 5 min + tbuffer
RELAXt = 2*tbuffer  # 15 min + tbuffer
PHOSPHENEt = 2*tbuffer  # 5 sec + tbuffer
timers_set, timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row, relax_sender, i, check1, check2, check3, check4, baselinedone, loop, boop, reward = reset_vars()
ccorr_at_laststimtime = 0
current_stim_time = 0
binary_matrix_stim = []
users = ['user1', 'user2', 'user3']
stims = ['motor', 'cognitive', 'relax', 'phosphene']
reportcols = ['stimulation time and date', 'Wmatrix', 'reward', 'sender', 'final binary stim matrix',
              'ccorr at stimulation time', 'ccorr after stim period']
report = pd.DataFrame(columns=reportcols)
avgccorraggregated = pd.DataFrame(columns=['avgccorr'])
finalstimvalue = []


# initialise W matrix
# there should be n rows for n inputs, m columns for m stims
Wmatrix = np.array([
    [0.5, 0.73, 0.5, 0.5],
    [0.5, 0.61, 0.5, 0.5],
    [0.6, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.6],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5]
])

#################################################################################################################

HEADER = 10
PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

usesynthetic = True
usersin = True
init = False
msg = []
user_inputs = ['undefined','undefined','undefined']
candidates = ['undefined','undefined','undefined']
connected = False


def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    global msg
    global user_inputs
    global connected
    global usesynthetic
    global candidates
    global usersin
    connected = True
    while connected:
        try:
            msg_length = json.loads(conn.recv(HEADER).decode(FORMAT))
            if msg_length:
                msg_length = int(msg_length)
                msg = json.loads(conn.recv(msg_length).decode(FORMAT))
                if msg["Username"] == "Spike":
                    ind = 0
                    print(user_inputs)
                    if not (user_inputs[0] == 'undefined' or user_inputs[1] == 'undefined' or user_inputs[2] == 'undefined'):
                        kaylas_angry() #one user runs the algorithm
                    user_inputs[0] = [msg['Binaries'][0], msg['eeg']]
                    if usesynthetic:
                        user_inputs[1] = user_inputs[0].copy()
                        user_inputs[2] = user_inputs[0].copy()
                if msg["Username"] == "Buffy":
                    user_inputs[1] = [msg['Binaries'][0], msg['eeg']]
                    ind = 1
                if msg["Username"] == "Willow":
                    user_inputs[2] = [msg['Binaries'][0], msg['eeg']]
                    ind = 2
                print(msg)
                #send off candidates to everyone if its ready
                if not (candidates[ind] == 'undefined'):
                    conn.send(json.dumps(candidates).encode(FORMAT))
                    print('sent candidate stim matrix to user')
                    candidates[ind] = 'undefined'
                else:
                    conn.send(json.dumps("no stims atm").encode(FORMAT))
                if msg == DISCONNECT_MESSAGE:
                    connected = False
        except Exception:
            pass

    conn.close()


def kaylas_angry():
    global init
    global usersin
    global user_inputs
    global usesynthetic
    global candidates
    global connected

    global cooldown
    global cooldown_time
    global saving
    global stim_threshold
    global reward_value
    global tbuffer
    global MOTORt
    global COGNITIVEt
    global RELAXt
    global PHOSPHENEt
    global users
    global stims
    global reportcols
    global report
    global avgccorraggregated
    global finalstimvalue
    global Wmatrix

    global timers_set
    global timer1
    global timer2
    global timer3
    global timer4
    global phos_row
    global phos_sender
    global motor_row
    global motor_sender
    global cog_row
    global cog_sender
    global relax_row
    global relax_sender
    global ccorr_at_laststimtime
    global current_stim_time
    global binary_matrix_stim
    global i
    global check1
    global check2
    global check3
    global check4
    global baselinedone
    global loop
    global boop
    global reward

    # init = True
    print('fuck you')
    avgccorr = compute_ccorr(np.asarray(user_inputs[0][1]),
                             np.asarray(user_inputs[1][1]),
                             np.asarray(user_inputs[2][1]))
    if saving:
        current_time = datetime.datetime.now(pytz.timezone('Australia/Sydney'))
        avgccorraggregated = server_save_data(current_time, avgccorr, avgccorraggregated)

    ################################## ALGORITHM CONT FROM CLIENT
    binary_matrix, BxW, candcolumnindex, candvalues, candidates, finalstimvalue, senderindex, senders \
        = do_Wmatrix_operations(np.asarray(user_inputs[0][0]),
                                np.asarray(user_inputs[1][0]),
                                np.asarray(user_inputs[2][0]),
                                Wmatrix,
                                nameofstims=stims,
                                nameofusers=users)
    finalstimvalue = stimulation_decision(finalstimvalue,
                                          stim_threshold,
                                          users,
                                          candidates,
                                          senders,
                                          cooldown)
    if (any(x == 1 for x in
            finalstimvalue) or timers_set):  # continue if there's actually a stim candidate, timers have been set, or one of our timers from a previous stim is finished,
        if not timers_set:
            current_stim_time = datetime.datetime.now(
                pytz.timezone('Australia/Sydney'))  # time stims were sent
            print('stims ready to send')#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # add cooldown of 30 mins to users being stimulated
            for bean in [0, 1, 2]:
                if finalstimvalue[bean] != 0:
                    cooldown[bean] += cooldown_time
            timers_set, timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row, relax_sender, binary_matrix_stim, ccorr_at_laststimtime \
                = set_timers(candidates, senderindex, binary_matrix, PHOSPHENEt, MOTORt, COGNITIVEt, RELAXt,
                             timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender,
                             cog_row,
                             cog_sender, relax_row, relax_sender, avgccorr)
        if timers_set and (
                check1 or check2 or check3 or check4):  # written like this in case there are multiple checks going at once, the keep function checks which timer is up and keeps the variable info for that specific stim
            BxW_colref, BxW_rowref, sender, timer1, timer2, timer3, timer4, check1, check2, check3, check4 = \
                keep_current_stim_data(timer1, timer2, timer3, timer4, check1, check2, check3, check4,
                                       phos_row,
                                       phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row,
                                       relax_sender)
            # computing circular correlation coefficient
            stim_ccorr = compute_ccorr(np.asarray(user_inputs[0][1]),
                                       np.asarray(user_inputs[1][1]),
                                       np.asarray(user_inputs[2][1]))
            ccorr_ers = (100 * (
                    stim_ccorr - ccorr_at_laststimtime) / ccorr_at_laststimtime)  # ers not erd because subtracting before from after
            if ccorr_ers > 0.5:
                reward = True
            else:
                reward = False
            print('ccorr_erd is ' + str(ccorr_ers))
            # if ccorr increased, successful state-stim associations should get rewarded
            Wmatrix = update_Wmatrix(reward,
                                     Wmatrix,
                                     BxW_rowref,
                                     BxW_colref,
                                     sender,
                                     binary_matrix_stim)
            print(Wmatrix)
            if saving:
                report = server_save_report(report, current_stim_time, Wmatrix, reward, users[sender],
                                            binary_matrix_stim, ccorr_at_laststimtime, stim_ccorr)
        if timer1 == timer2 == timer3 == timer4 == -1:
            print('ccorrperiod donezo')
            # is there anything else to reset???????????????????????????????????????
            timers_set, timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row, relax_sender, i, check1, check2, check3, check4, baselinedone, loop, boop, reward = reset_vars()
            print("IT FUCKING WORKED")
    user1 = False
    user2 = False
    user3 = False
    # time.sleep(1)     # cooldowns & timers updated every second
    # kaylas_angry()

    timer1, timer2, timer3, timer4, check1, check2, check3, check4 = update_timers(timers_set, timer1,
                                                                                   timer2,
                                                                                   timer3, timer4, check1,
                                                                                   check2, check3, check4)
    cooldown = update_cooldowns(cooldown)
    print(cooldown)

##########################################################################END ALGORITHM


def start():
    global candidates
    global connected
    global user_inputs
    global init
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        #this loop DOES end you cant put stuff in here
start()
print("[STARTING] server is starting...")

