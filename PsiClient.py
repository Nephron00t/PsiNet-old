# import RPi.GPIO as GPIO #relay for controlling pi
# import bluepy.btle as btle #needs to be linux
import numpy as np
from collections import OrderedDict
import json
import time
from datetime import datetime
import socket
import json
import time
import numpy as np
import pandas as pd
import datetime
import pytz
import time
import brainflow
import numpy as np
import pandas as pd
import threading
import matplotlib
matplotlib.use('Agg')
from copy import copy
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from numpy.fft import fft, ifft
from astropy.stats import circcorrcoef
from enum import Enum
from functions import get_psd, get_erd, get_binary_decision, client_save_data, \
    stimulation_decision, compute_ccorr, update_Wmatrix, do_Wmatrix_operations, set_timers, keep_current_stim_data, \
    update_timers, update_cooldowns, server_save_data, server_save_report, reset_vars  # THESE ARE MY MADE UP FUNCTIONS, it needs to have "functions" file in same folder as directory




saving = False
usesynthetic = False
oneuser = True

######################################## STIM BLUETOOTH STUFF, specific to each device

    # C3448107-3C42-4C63-9CFB-5B1702CBE0F6: foc.us v3
    # S/N : 78A50458B8DA
    # MAC: 78:A5:04:58:B8:DA
    # FCC ID: 2AAH6NUFC9
    # 0000AAB0-F845-40FA-995D-658A43FEEA4C
    ############################################################################
    ####################### DEFAULT PROGRAMS FROM MANUAL #######################
    ########### RIPPLE | GAMER | eDREAM | Noise | Pulse | 3rd eye ##############
    ############################################################################
    ############################ PSINET PROGRAMS ###############################
    #   X is target Y is reference
    #   Anodal = X is anode
    #
    #   STIM            CODE            TYPE        mA      Time        Hz
    #   TeamFlow:       1 (RIPPLE)      tACS        1mA     2mins       40hz
    #   Cognitive:      2 (3rdEYE)      tDCS        2mA     5mins       -
    #   MotorSync:      3 (WAVE)        tACS        1mA     2mins       6hz
    #   Phosphene:      4 (PULSE)       tACS        1.5mA   5sc         12hz
    #   Motor:          5 (NOISE)       tDCS        1.5mA   2mins       -
    #   Relax:          6 (EDREAM)      tDCS        2mA     15mins      -
#
# GPIO.setmode(GPIO.BCM) #start board
# GPIO.setup(23,GPIO.OUT) #settin up pins
# GPIO.setup(24,GPIO.OUT)
# # STIM TYPE AS STRING, FROM
# focus1 = btle.Peripheral("B4:99:4C:4F:88:84") #specific for each focus device, different for each user
# service1 = focus1.getServiceByUUID("0000AAB0-F845-40FA-995D-658A43FEEA4C")
# characteristic1 = service1.getCharacteristics()[0]
#
#
# def stimtypes(zap):
#     global Stim_Motor
#     global Stim_Cognitive
#     global Stim_Phosphene
#     global Stim_Relax
#     stim = 'none' #in case it came in as none already
#     zap == 'cognitive'
#     if zap == 'motor':
#         GPIO.output(23,GPIO.LOW)
#         GPIO.output(24,GPIO.LOW)
#         Stim_Motor = characteristic1.write((bytes([2, 7, 5, 0, 0, 0])))
#     elif zap == 'cognitive':
#         GPIO.output(23,GPIO.HIGH)
#         GPIO.output(24,GPIO.HIGH)
#         Stim_Cognitive = characteristic1.write((bytes([2, 7, 2, 0, 0, 0])))
#     elif zap == 'phosphene':
#         GPIO.output(23,GPIO.HIGH)
#         GPIO.output(24,GPIO.HIGH)
#         Stim_Phosphene = characteristic1.write((bytes([2, 7, 4, 0, 0, 0])))
#         time.sleep(2)
#         Stim_Phosphene = characteristic1.write((bytes([2, 7, 4, 0, 0, 0])))
#     elif zap == 'relax':
#         GPIO.output(23,GPIO.HIGH)
#         GPIO.output(24,GPIO.LOW)
#         Stim_Relax = characteristic1.write((bytes([2, 7, 6, 0, 0, 0])))
#     return

######################################## NETWORKING VARS FOR CLIENT
HEADERSIZE = 10   # MAKES AN ABIRTAILY LONG SPACE AT THE START OF ANY MESSAGE YOU CAN CONTAIN METADATA IN
PORT = 8080       # THE SERVER PORT THE MESSAGE IS BEING SENT TO
FORMAT = 'utf-8'  # ENCODING FORMAT
# SERVER = "14.137.219.228"   #SERVER IP ADDRESS (FOR THE BEAST AT HAWTHORN)
SERVER = '192.168.0.24'

ADDR = (SERVER, PORT)       # TUPLE OF SERVER IP AND PORT NUMBERS

my_username = "Willow"   # CLIENT USERNAME (SPECIFIC TO THE RASPBERRY PI BEING USED)
my_user_index = 2 #spike is like 'user 1' so has 0

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    #CALL THIS IF YOU NEED TO CLIENT RELATED STUFF
client.connect(ADDR)   # E.G. LIKE HERE, WHERE WE CONNECT TO THE SERVER AS CLIENT

############################### NETWORKING SEND FUNCTION
stims_from_server = []
stim = 'none'
zippidy = False
def send(msg):
    global zippidy
    global client
    global FORMAT
    global HEADERSIZE
    # THIS IS THE FUNCTION FOR SENDING MESSAGE
    message = msg.encode(FORMAT)          # WE TAKE THE MESSAGE AND ENCODE IT AS UTF-8 BYTES
    msg_length = len(message)
    print(msg_length)# WE GET THE LENGTH OF THAT
    send_length = str(msg_length).encode(FORMAT)   # WE ENCODE THE LENGTH AS A STRING
    send_length += b' ' * (HEADERSIZE - len(send_length))  # PUT THE LENGTH INSIDE THE HEADDER
    client.sendall(send_length)     # SEND THE MESSAGE LENGTH TO THE SERVER SO IT KNOWS HOW LONG THE MESSAGE IS GONNA BE
    client.sendall(message)       # THEN SEND THE MESSAGE

def getstims():
    time.sleep(3)
    global zippidy
    global FORMAT
    global client
    global stims_from_server
    incoming_msg = json.loads(client.recv(50).decode(FORMAT))
    print(incoming_msg) # RECEIVE MESSAGES FROM THE SERVER IF IT HAS ANY.
    if not (incoming_msg == "no stims atm"):
        stims_from_server = np.asarray(incoming_msg)
        zippidy = True
    else:
        zippidy = False

########################################################

BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
if usesynthetic:
    board_id = BoardIds.SYNTHETIC_BOARD.value
else:
    # board_id = BoardIds.CYTON_DAISY_BOARD.value
#     params.serial_port = "/dev/ttyUSB0"
#     params.timeout = 100
# #     params.serial_number = "DO015QPL"
    board_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
    params.ip_address = '192.168.0.33' #ip of wifi shield
    params.ip_port = 8090

board = BoardShim(board_id, params)
sampling_rate = BoardShim.get_sampling_rate(board_id)
board.prepare_session()
board.start_stream()
is_streaming = True
BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
eeg_channels = BoardShim.get_eeg_channels(board_id)  # gets structure for indexing dataframes
erdcols = ['cog_theta_f1f2f7f8f3f4',
           'cog_alpha_o1o2p3p4',
           'conc_theta_f1f2',
           'conc_beta_f1f2',
           'conc_theta_c3c4o2',
           'conc_beta_c3c4o2',
           'motor_mu_c3c4cz',
           'dVal_alphaf4/betaf4-alphaf3/betaf3',
           'dAr_log2betaFz/alphaFz'
           ]
rows = ['mu', 'theta', 'alpha', 'beta']
inputs = ['cognitive load', 'concentration', 'motor imagery', 'frustration', 'excitement', 'meditation', 'boredom']
erd = pd.DataFrame(columns=erdcols)
bands = pd.DataFrame(columns=eeg_channels, index=rows)
binary = pd.DataFrame(columns=inputs)
binaryaggregated = binary.copy()
binary_matrix = []

#initialising some dfs for saving
filteredaggregated = pd.DataFrame(columns=eeg_channels)
erdaggregated = pd.DataFrame(columns=erdcols)
binaryaggregated = pd.DataFrame(columns=inputs)
timers_set,timer1,timer2,timer3,timer4,phos_row,phos_sender,motor_row,motor_sender,cog_row,cog_sender,relax_row,relax_sender,i,check1,check2,check3,check4,baselinedone,loop,boop,reward = reset_vars()

###################################################################START###############################################

while is_streaming:  # while the stream is on (true)
    # brainflow streaming stuff
    time.sleep(5)
    data = board.get_board_data() # get all available data from the board
    # eeg_data = data[eeg_channels, :]  #grabs the eeg channel data from 'data'
    # new_data = pd.DataFrame(np.transpose(eeg_data), columns=eeg_channels)
    # df = df.append(new_data, ignore_index=True)

    for channel in eeg_channels:
        # bandpass for 5-50Hz: note 4th order bessel bandpass filter
        DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(board_id), 26.5, 21.5, 4,
                                    FilterTypes.BESSEL.value, 0)
        # 3rd order buttwerowth bandstop filter - this takes out a weird bluetooth caused spike at around 24.5Hz:
        DataFilter.perform_bandstop(data[channel], BoardShim.get_sampling_rate(board_id), 24.25, 1.0, 3,
                                    FilterTypes.BUTTERWORTH.value, 0)
        # de-noising filter
        try: #we are doing this because we kept getting errors, the idea is that if it does get an error, this round will be skipped and the last round of data will be used for further calculations
            DataFilter.perform_wavelet_denoising(data[channel], 'coif3', 3)

            #calculating power spectrum values using Welch's method
            bands = get_psd(data,
                            channel,
                            nfft,
                            sampling_rate,
                            window=WindowFunctions.BLACKMAN_HARRIS.value,
                            output_df=bands)
        except Exception:
            pass

        # down_sampling for sending to server to do ccorr. this downsampled data is not used for erd calculations
        if not boop:
            eeg_data = [DataFilter.perform_downsampling(data[channel], 125, AggOperations.MEDIAN.value)]
        else:
            eeg_data = np.concatenate(
                (eeg_data, [DataFilter.perform_downsampling(data[channel], 125, AggOperations.MEDIAN.value)]))

        # getting power spectrum values for mu, theta, alpha and beta bands (note freq start & freq-end values) using welch technique, for each channel

        loop = True
        boop = True
    boop = False

    if baselinedone == False:
        baseline = bands.copy()
        baselinedone = True
        print(np.transpose(baseline))
        print('baseline values calculated, now calculating erd')
        loop = False
    elif baselinedone and loop:
        erd = get_erd(output=erd,
                      current_psd=bands,
                      baseline_psd=baseline,
                      F1=1, F2=2, F3=9, F4=10, F7=11, F8=12, O1=7, O2=8, P3=15, P4=16, C3=3, C4=4, Fz=13, Cz=14)
        binary = get_binary_decision(output=binary,
                                     erd=erd,
                                     stims_threshold=0.5,
                                     emotion_threshold=0.5)
        print(binary.transpose())

        if saving:
            current_time = datetime.datetime.now(pytz.timezone('Australia/Sydney'))
            filteredaggregated, erdaggregated, binaryaggregated \
                = client_save_data(current_time,
                                   filtered_data=data,
                                   filteredaggregated=filteredaggregated,
                                   erd_data=erd,
                                   erdaggregated=erdaggregated,
                                   binary_data=binary,
                                   binaryaggregated=binaryaggregated)

        #PACKAGE UP THE RAW DATA AS A NICE LIL ROUNDED LIST AND SEND IT
        eeg_round = np.round(eeg_data, 1)
        eeg_list = eeg_round.tolist()
        binary_list = binary.values.tolist()
        package = {"eeg": eeg_list,
                   "Binaries": binary_list,
                   "Username": my_username}
        send(json.dumps(package))
        print('sent package to server')
        # Get the  lil stimulations from the stupid lil server and zap the braiiinnn
        get_stims = threading.Thread(target=getstims)
        get_stims.start()

        if zippidy:
            zap = stims_from_server[my_user_index]
            if (oneuser == False) and not (zap == 'none'):
                # stimtypes(zap)#ie execute the tES stimulation
                print('i got ' + zap + 'stimulation')
            zippidy = False
