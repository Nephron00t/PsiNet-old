import time
import numpy as np
import pandas as pd
import datetime
import pytz
import time
import pandas as pd
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
#import bluepy.btle as btle #needs to be linux
#import RPi.GPIO as GPIO #relay for controlling pi
matplotlib.use('Agg')
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from numpy.fft import fft, ifft
from astropy.stats import circcorrcoef
from enum import Enum

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations


def stimtypes(candidatestim):
    stim = 'none' #in case it came in as none already
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
    #
    # GPIO.setmode(GPIO.BOARD) #start board
    # GPIO.output(14,GPIO.OUT) #settin up pins
    # GPIO.output(15,GPIO.OUT)
    #
    #
    # # STIM TYPE AS STRING, FROM
    # focus1 = btle.Peripheral("78:A5:04:58:B8:DA") #specific for each focus device, different for each user
    # service1 = focus1.getServiceByUUID("0000AAB0-F845-40FA-995D-658A43FEEA4C")
    # characteristic1 = service1.getCharacteristics()[0]
    # # Stim_MotorSync = characteristic1.write((bytes([2, 7, 3, 0, 0, 0])))  # not using atm
    # # Stim_TeamFlow = characteristic1.write((bytes([2, 7, 1, 0, 0, 0])))  # not using atm
    # Stim_Motor = characteristic1.write((bytes([2, 7, 5, 0, 0, 0])))
    # Stim_Cognitive = characteristic1.write((bytes([2, 7, 2, 0, 0, 0])))
    # Stim_Phosphene = characteristic1.write((bytes([2, 7, 4, 0, 0, 0])))
    # Stim_Relax = characteristic1.write((bytes([2, 7, 6, 0, 0, 0])))
    #
    # if candidatestim == 'motor':
    #     GPIO.output(14,GPIO.LOW)
    #     GPIO.output(15,GPIO.LOW)
    #     stim = Stim_Motor
    # elif candidatestim == 'cognitive':
    #     GPIO.output(14,GPIO.HIGH)
    #     GPIO.output(15,GPIO.HIGH)
    #     stim = Stim_Cognitive
    # elif candidatestim == 'phosphene':
    #     GPIO.output(14,GPIO.HIGH)
    #     GPIO.output(15,GPIO.HIGH)
    #     stim = Stim_Phosphene
    # elif candidatestim == 'relax':
    #     GPIO.output(14,GPIO.HIGH)
    #     GPIO.output(15,GPIO.LOW)
    #     stim = Stim_Relax
    return stim



########################################FUNCTIONS#############################################################
def reset_vars():
    timers_set = False
    timer1 = -5
    timer2 = -5
    timer3 = -5
    timer4 = -5
    phos_row = -1
    phos_sender = -1
    motor_row = -1
    motor_sender = -1
    cog_row = -1
    cog_sender = -1
    relax_row = -1
    relax_sender = -1
    i = 0
    check1 = False
    check2 = False
    check3 = False
    check4 = False
    baselinedone = False
    loop = True
    boop = False
    reward = False
    return timers_set,timer1,timer2,timer3,timer4,phos_row,phos_sender,motor_row,motor_sender,cog_row,cog_sender,relax_row,relax_sender,i,check1,check2,check3,check4,baselinedone,loop,boop,reward

def get_psd(input_df, channel, nfft, sampling_rate, window, output_df):
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
    data=input_df
    bands=output_df

    psd = DataFilter.get_psd_welch(data=data[channel],
                                   nfft=nfft,  # FFT size
                                   overlap=nfft // 2,  # FFT window overlap, must be between 0 and nfft
                                   sampling_rate=sampling_rate,
                                   window=WindowFunctions.BLACKMAN_HARRIS.value)

    band_power_mu = DataFilter.get_band_power(psd, freq_start=9.0, freq_end=11.0)
    band_power_theta = DataFilter.get_band_power(psd, freq_start=4.0, freq_end=8.0)
    band_power_alpha = DataFilter.get_band_power(psd, freq_start=8.0, freq_end=12.5)
    band_power_beta = DataFilter.get_band_power(psd, freq_start=12.5, freq_end=30.0)
    bands.at['mu', channel] = band_power_mu
    bands.at['theta', channel] = band_power_theta
    bands.at['alpha', channel] = band_power_alpha
    bands.at['beta', channel] = band_power_beta
    return bands


def get_erd(output, current_psd, baseline_psd, F1, F2, F3, F4, F7, F8, O1, O2, P3, P4, C3, C4, Fz, Cz):
    import numpy as np
    erd = output
    bands = current_psd
    baseline = baseline_psd

    # cognitive load [1,2,9,10,11,12][7,8,15,16]
    # print(np.transpose(bands.astype(float).round(2)))
    cog_theta_baseline = baseline.at['theta', F1] + baseline.at['theta', F2] + baseline.at['theta', F7] + baseline.at[
        'theta', F8] + baseline.at['theta', F3] + baseline.at['theta', F4]
    cog_theta_current = bands.at['theta', F1] + bands.at['theta', F2] + bands.at['theta', F7] + bands.at['theta', F8] + \
                        bands.at['theta', F3] + bands.at['theta', F4]
    erd['cog_theta_f1f2f7f8f3f4'] = [(100 * (cog_theta_baseline - cog_theta_current) / cog_theta_baseline)]
    cog_alpha_baseline = baseline.at['alpha', O1] + baseline.at['alpha', O2] + baseline.at['alpha', P3] + baseline.at[
        'alpha', P4]
    cog_alpha_current = bands.at['alpha', O1] + bands.at['alpha', O2] + bands.at['alpha', P3] + bands.at['alpha', P4]
    erd['cog_alpha_o1o2p3p4'] = [(100 * (cog_alpha_baseline - cog_alpha_current) / cog_alpha_baseline)]

    # concentration[1,2][3,4,8]
    conc_theta_baseline = baseline.at['theta', F1] + baseline.at['theta', F2]
    conc_theta_current = bands.at['theta', F1] + bands.at['theta', F2]
    erd['conc_theta_f1f2'] = [
        100 * (conc_theta_baseline - conc_theta_current) / (baseline.at['theta', F1] + baseline.at['theta', F2])]

    conc_beta_baseline = baseline.at['beta', F1] + baseline.at['beta', F2]
    conc_beta_current = bands.at['beta', F1] + bands.at['beta', F2]
    erd['conc_beta_f1f2'] = [(100 * (conc_beta_baseline - conc_beta_current) / conc_beta_baseline)]

    conc_theta_baseline = baseline.at['theta', C3] + baseline.at['theta', C4] + baseline.at['theta', O2]
    conc_theta_current = bands.at['theta', C3] + bands.at['theta', C4] + baseline.at['theta', O2]
    erd['conc_theta_c3c4o2'] = [(100 * (conc_theta_baseline - conc_theta_current) / conc_theta_baseline)]

    conc_beta_baseline = baseline.at['theta', C3] + baseline.at['theta', C4] + baseline.at['theta', O2]
    cocn_beta_current = bands.at['theta', C3] + bands.at['theta', C4] + baseline.at['theta', O2]
    erd['conc_beta_c3c4o2'] = [100 * ((conc_beta_baseline - cocn_beta_current) / conc_beta_baseline)]

    # motor[3,4]
    erd['motor_mu_c3c4cz'] = [
        100 * ((baseline.at['mu', C3] + baseline.at['mu', C4] + baseline.at['mu', Cz]) - (bands.at['mu', C3] + bands.at['mu', C4] + bands.at['mu', Cz])) / (
                    baseline.at['mu', C3] + baseline.at['mu', C4] + baseline.at['mu', Cz])]

    # dVal and dAr use F3,F4,Fz
    baseline_dVal = (baseline.at['alpha', F4] / baseline.at['beta', F4]) - (
                baseline.at['alpha', F3] / baseline.at['beta', F3])
    baseline_dAr = np.log2((baseline.at['beta', Fz] / baseline.at['alpha', Fz]))
    present_dVal = (bands.at['alpha', F4] / bands.at['beta', F4]) - (
                bands.at['alpha', F3] / bands.at['beta', F3])
    present_dAr = np.log2((baseline.at['beta', Fz] / baseline.at['alpha', Fz]))
    erd['dVal_alphaf4/betaf4-alphaf3/betaf3'] = [100 * (baseline_dVal - present_dVal) / baseline_dVal]
    erd['dAr_log2betaFz/alphaFz'] = [100 * (baseline_dAr - present_dAr) / baseline_dAr]

    return erd


def get_binary_decision(output, erd, stims_threshold, emotion_threshold):
    binary = output
    erd = erd
    t = stims_threshold
    te = emotion_threshold
    if (erd.iat[0, 0] <= -t) and (erd.iat[0, 1] >= t):
        # with high cognitive load, we expect theta to increase, so negative (erd), we expect alpha to decrease, so positive (ERD)
        binary['cognitive load'] = [1]
    else:
        binary['cognitive load'] = [0]
    if (erd.iat[0, 2] <= -t) and (erd.iat[0, 3] <= -t) and (erd.iat[0, 4] >= t) and (erd.iat[0, 5] >= t):
        # with high concentration, we expect theta and beta to increase in frontal, so negative (erd). we expect theta and beta to decrease in c3c4o2, so positive (ERD)
        binary['concentration'] = [1]
    else:
        binary['concentration'] = [0]
    if erd.iat[0, 6] >= t:
        # when moving or imagining moving, we expect mu to decrease in motor cortex, so positive (ERD)
        binary['motor imagery'] = [1]
    else:
        binary['motor imagery'] = [0]
    if erd.iat[0, 7] <= -te and erd.iat[0, 8] >= te:
        binary['frustration'] = [1]
    else:
        binary['frustration'] = [0]
    if erd.iat[0, 7] >= te and erd.iat[0, 8] >= te:
        binary['excitement'] = [1]
    else:
        binary['excitement'] = [0]
    if erd.iat[0, 7] >= te and erd.iat[0, 8] <= -te:
        binary['meditation'] = [1]
    else:
        binary['meditation'] = [0]
    if erd.iat[0, 7] <= -te and erd.iat[0, 8] <= -te:
        binary['boredom'] = [1]
    else:
        binary['boredom'] = [0]
    return binary


def do_Wmatrix_operations(u1_binary, u2_binary, u3_binary, Wmatrix, nameofstims, nameofusers):
    stims = nameofstims
    users = nameofusers

    binary_matrix = np.array([u1_binary, u2_binary, u3_binary])
    binary_matrix = binary_matrix.reshape(3, u1_binary.size)  # THIS CODE S THUS SPECIFIC FOR THREE USERS

    # start working out if we're going to stimulate
    BxW = binary_matrix.dot(
        Wmatrix)  # result = 7 x 4 matrix, where max of each row n is the nth user's candidate stimulation
    candcolumnindex = BxW.argmax(
        axis=1)  # result is array where nth value is index (corresponding to stims) of nth user's candidate stim
    candvalues = BxW.max(axis=1)
    candidates = [stims[candcolumnindex[0]], stims[candcolumnindex[1]],
                  stims[candcolumnindex[2]]]  # actual names of stim candidates
    D = np.array([candvalues, candvalues, candvalues])  # repeat candvalues for n users, so each row represents a user
    np.fill_diagonal(D, 0)  # so that users cannot stimulate themselves
    finalstimvalue = D.max(axis=1)
    senderindex = D.argmax(axis=1)  # index of stimulation sender
    senders = [users[senderindex[0]], users[senderindex[1]],
               users[senderindex[2]]]  # names of senders, from users array
    return binary_matrix, BxW, candcolumnindex, candvalues, candidates, finalstimvalue, senderindex, senders


def stimulation_decision(finalstimvalue, stim_threshold, users, candidates, senders, cooldown):
    for i in [0, 1, 2]:
        if finalstimvalue[i] > stim_threshold and cooldown[i] == 0:
            finalstimvalue[i] = 1
            print(users[i] + ' receives ' + candidates[i] + ' stimulation from ' + senders[i])
        else:
            finalstimvalue[i] = 0
            candidates[i] = 'none'
            print(users[i] + ' receives ' + candidates[i] + ' stimulation')
    return finalstimvalue


def compute_ccorr(u1_eeg_data, u2_eeg_data, u3_eeg_data):
    # MAKE SURE ALL EEG DATA ARE THE SAME LENGTH
    u1_resized = []
    u2_resized = []
    u3_resized = []
    for channel in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]: #for each string
        l1 = len(u1_eeg_data[channel])
        l2 = len(u2_eeg_data[channel])
        l3 = len(u3_eeg_data[channel])
        if l1 != l2 or l1 != l3 or l2 != l3:
            minlen = min(l1, l2, l3) #making all the channels equal to the min length between users for each channel
            u1_resized.append(u1_eeg_data[channel][0:minlen])
            u2_resized.append(u2_eeg_data[channel][0:minlen])
            u3_resized.append(u3_eeg_data[channel][0:minlen])

    ccorr1 = circcorrcoef(np.asarray(u1_resized), np.asarray(u2_resized))
    ccorr2 = circcorrcoef(np.asarray(u1_resized), np.asarray(u2_resized))
    ccorr3 = circcorrcoef(np.asarray(u3_resized), np.asarray(u3_resized))
    if ccorr1 == 1 or ccorr2 == 1 or ccorr3 == 1:  # just so that we can never get an undefined arctanh
        if ccorr1 == 1:
            ccorr1 = 0.99
        if ccorr2 == 1:
            ccorr2 = 0.99
        if ccorr3 == 1:
            ccorr3 = 0.99
    # transforming to fisher's Z to calculate average correlation, since correlations themselves should not be averaged since they are cosines and these cant be aggregated
    fishersz1 = np.arctanh([ccorr1])
    fishersz2 = np.arctanh([ccorr2])
    fishersz3 = np.arctanh([ccorr3])
    fishersz = np.array([fishersz1, fishersz2, fishersz3])
    fzmean = fishersz.mean()
    avgccorr = np.tanh(fzmean)  # scalar, not in []
    return avgccorr


def update_Wmatrix(reward, Wmatrix, BxW_rowref, BxW_colref, sender, binary_matrix_stim):
    binm = binary_matrix_stim[sender]
    if reward and Wmatrix[BxW_rowref, BxW_colref] < 1:
        for r in [0, 1, 2, 3, 4, 5, 6]:
            Wmatrix[r, BxW_colref] += 0.05 * binm[r]
    if not reward and Wmatrix[BxW_rowref, BxW_colref] > 0:
        for r in [0, 1, 2, 3, 4, 5, 6]:
            Wmatrix[r, BxW_colref] -= 0.05 * binm[r]
    return Wmatrix


def set_timers(candidates, senderindex, binary_matrix, PHOSPHENEt, MOTORt, COGNITIVEt, RELAXt, timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row,relax_sender, avgccorr):
    binary_matrix_stim = binary_matrix
    for i in [0, 1, 2]:
        if candidates[i] == 'phosphene':
            timer1 = PHOSPHENEt
            phos_row = i
            phos_sender = senderindex[i]
        elif candidates[i] == 'motor':
            timer2 = MOTORt
            motor_row = i
            motor_sender = senderindex[i]
        elif candidates[i] == 'cognitive':
            timer3 = COGNITIVEt
            cog_row = i
            cog_sender = senderindex[i]
        elif candidates[i] == 'relax':
            timer4 = RELAXt
            relax_row = i
            relax_sender = senderindex[i]
    timers_set = True
    ccorr_at_laststimtime = avgccorr
    return timers_set, timer1, timer2, timer3, timer4, phos_row, phos_sender, motor_row, motor_sender, cog_row, cog_sender, relax_row, relax_sender, binary_matrix_stim, ccorr_at_laststimtime


def keep_current_stim_data(timer1, timer2, timer3, timer4, check1, check2, check3, check4, phos_row, phos_sender, motor_row, motor_sender, cog_row,
                           cog_sender, relax_row, relax_sender):
    if check1:
        BxW_colref = 3
        BxW_rowref = phos_row
        sender = phos_sender
        timer1 = -5
        check1 = False
    elif check2:
        BxW_colref = 0
        BxW_rowref = motor_row
        sender = motor_sender
        timer2 = -5
        check2 = False
    elif check3:
        BxW_colref = 1
        BxW_rowref = cog_row
        sender = cog_sender
        timer3 = -5
        check3 = False
    elif check4:
        BxW_colref = 2
        BxW_rowref = relax_row
        sender = relax_sender
        timer4 = -5
        check4 = False
    return BxW_colref, BxW_rowref, sender, timer1, timer2, timer3, timer4, check1, check2, check3, check4


def client_save_data(current_time,filtered_data,filteredaggregated,erd_data,erdaggregated, binary_data,binaryaggregated):
    # time = current_time
    # erd = erd_data
    # binary = binary_data
    # hour = str(current_time.hour).replace(":","x")
    #
    # filtered_data = pd.DataFrame(np.transpose(filtered_data))
    # filteredaggregated.loc[len(filteredaggregated), 'time'] = time
    # filteredaggregated = filteredaggregated.append(filtered_data, ignore_index=True)
    # filteredaggregated.to_csv(r'/home/pi/Desktop/datafiles/filtereddata'+hour+'.csv')
    #
    # erdaggregated.loc[len(erdaggregated), 'time'] = time
    # erdaggregated = erdaggregated.append(erd, ignore_index=True)
    # erdaggregated.to_csv(r'/home/pi/Desktop/datafiles/erddata'+hour+'.csv')
    #
    # binaryaggregated.loc[len(binaryaggregated), 'time'] = time
    # binaryaggregated = binaryaggregated.append(binary, ignore_index=True)
    # binaryaggregated.to_csv(r'/home/pi/Desktop/datafiles/binarydata'+hour+'.csv')
    return filteredaggregated, erdaggregated, binaryaggregated


def server_save_data(current_time,avgccorr,avgccorraggregated):
    # hour = str(current_time.hour).replace(":", "x")
    # time = current_time
    # avgccorraggregated.loc[len(avgccorraggregated), 'time'] = time
    # avgccorr = pd.DataFrame([avgccorr], columns=['avgccorr'])
    # avgccorraggregated = avgccorraggregated.append(avgccorr, ignore_index=True)
    # avgccorraggregated.to_csv(r'C:\PsiNet\datafiles\ccorrdata'+hour+'.csv')
    return avgccorraggregated


def server_save_report(report, current_stim_time, Wmatrix, reward, sender, binary_matrix_stim, ccorr_at_laststimtime, stim_ccorr):
    current_time = datetime.datetime.now(pytz.timezone('Australia/Sydney'))
    hour = str(current_time.hour).replace(":", "x")
    newreport = [current_stim_time,Wmatrix,reward,sender,binary_matrix_stim,ccorr_at_laststimtime,stim_ccorr]
    report.loc[len(report)] = newreport
    report.to_csv(r'C:\PsiNet\datafiles\stimreports'+hour+'.csv')
    return report


def update_timers(timers_set, timer1, timer2, timer3, timer4, check1, check2, check3, check4):
    if timers_set:
        if timer1 > -1:
            if timer1 == 0:
                check1 = True
            timer1 -= 5
        if timer2 > -1:
            if timer2 == 0:
                check2 = True
            timer2 -= 5
        if timer3 > -1:
            if timer3 == 0:
                check3 = True
            timer3 -= 5
        if timer4 > -1:
            if timer4 == 0:
                check4 = True
            timer4 -= 5
    return timer1, timer2, timer3, timer4, check1, check2, check3, check4


def update_cooldowns(cooldown):
    for cool in [0, 1, 2]:
        if cooldown[cool] != 0:
            cooldown[cool] -= 5
        if cooldown[cool] < 0:
            cooldown[cool] = 0
    return cooldown