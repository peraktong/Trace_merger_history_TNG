import deepdish as dd
import illustris_python as il
import math
import numpy as np
import gzip
import pickle
import sys
from scipy import integrate
import time, random
import h5py
import threading
import os.path
import _pickle as cpickle
from scipy import spatial
from multiprocessing import Process
import multiprocessing
import csv


def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return np.nan


def box_smooth(data_array):
    N = len(data_array)

    data_smooth = []

    for i in range(0, N):
        data_i = data_array[int(np.maximum(i - 1, 0)):int(np.minimum(i + 2, N))]
        # print(np.nanmean(data_i))

        data_smooth.append(np.nanmean(data_i))

    data_smooth = np.array(data_smooth).ravel()
    data_smooth[0] = np.nanmedian(data_array[:1])
    return data_smooth


def bootstrap_scatter_err(samples):
    mask_finite = np.isfinite(samples)
    samples = samples[mask_finite]
    index_all = range(len(samples))
    err_all = []
    N = 100
    for i in range(0, N):
        index_choose = np.random.randint(0, len(samples) - 1, len(samples))
        # k_i = np.nanstd(samples[index_choose])
        k_i = np.percentile(samples[index_choose], 84) - np.percentile(samples[index_choose], 16)
        k_i = k_i / 2
        err_all.append(k_i)
    err_all = np.array(err_all)

    return err_all


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf


def Mpeak_log_to_Vpeak_log(Mpeak_log):
    return 0.3349 * Mpeak_log - 1.672


G = 4.301 * 10 ** (-9)
cons = (4 * G * np.pi / (3 * (1 / 24 / (1.5 * 10 ** (11))) ** (1 / 3))) ** 0.5


def calculate_v_dispersion(Mh):
    return Mh ** (1 / 3) * cons


exp = np.vectorize(exp)
log10 = np.vectorize(log10)

## redshift reading:

# read redshift:

# detect server or local automatically:
if os.path.isdir("/Volumes/SSHD_2TB") == True:
    print("The code is on Spear of Adun")
    data_path = "/Volumes/Data_10TB/"

# sirocco2!!
elif os.path.isdir("/mount/sirocco1/jc6933/test") == True:
    data_path = "/mount/sirocco2/jc6933/Data_sirocco/"
    print("The code is on Sirocco")
else:
    print("The code is on local")
    data_path = "/Volumes/Extreme_SSD/Data/"

print("data_path %s" % data_path)


N_snap_shot = []
N_redshift = []

with open(data_path+"TNG100/TNG100-1/url/" + "TNG100_information_csv.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for i, row in enumerate(csv_reader):
        if i > 2:
            N_snap_shot.append(row[1])
            N_redshift.append(row[2])

N_snap_shot = np.array(N_snap_shot, dtype=int)
N_redshift = np.array(N_redshift, dtype=float)


Mh_target_log = np.linspace(10, 13, 18)

##!! extend the Mh_target a little bit but with bigger bin size for high Mh:
Mh_target_log_high = np.linspace(13, 14.5, 7)

Mh_target_log = np.append(Mh_target_log, Mh_target_log_high)

binsize = 0.05

## define a function which returns scatter at input redshift:

n_group = 99

### read the group catalog:
fields = ['GroupBHMass', "Group_M_Crit200", "GroupFirstSub"]

basePath = data_path + 'TNG100/TNG100-1/output'
halos = il.groupcat.loadHalos(basePath, n_group, fields=fields)
halos.keys()

GroupBHMass = halos['GroupBHMass'] * 1e10 / 0.704
Group_M_Crit200 = halos["Group_M_Crit200"] * 1e10 / 0.704
GroupFirstSub = halos["GroupFirstSub"]

### for subhalo
fields = ['SubhaloMass', 'SubhaloSFRinRad', "SubhaloVmax", "SubhaloStellarPhotometricsMassInRad", "SubhaloBHMass",
          "SubhaloGrNr"]

basePath = data_path + 'TNG100/TNG100-1/output'
subhalos = il.groupcat.loadSubhalos(basePath, n_group, fields=fields)
subhalos.keys()

mass_msun = subhalos['SubhaloMass'] * 1e10 / 0.704
Vmax = subhalos['SubhaloVmax']
SubhaloStellarPhotometricsMassInRad = subhalos['SubhaloStellarPhotometricsMassInRad'] * 1e10 / 0.704
SubhaloBHMass = subhalos["SubhaloBHMass"]
SubhaloGrNr = subhalos["SubhaloGrNr"]

# plot and check:

Vmax_halo = Vmax[GroupFirstSub]
Vmax_halo_log = log10(Vmax_halo)
Ms_halo = SubhaloStellarPhotometricsMassInRad[GroupFirstSub]
Mh_first_sub = mass_msun[GroupFirstSub]


fields = ['SubhaloMass', 'SubfindID', 'SnapNum', 'SubhaloStellarPhotometricsMassInRad', 'SubhaloVmax',"SubhaloBHMass"]

# save path:
# data_path + "TNG100/" + "TNG100-1/scatter/"
hf = h5py.File("Trace_merger_TNG100_v1.h5", "w")
N_spike_array = []

speed = 1
time_previous = 0

## !! Here N_tot means you trace the history for the first N_tot halos
N_tot = 20000

for i in range(0,N_tot):

    if i % 1000 == 0:
        speed = (1000 / (time.time() - time_previous))

        print("sample per second=%.2f" % speed)

        print("Doing %d of %d, time left=%.2f seconds" % (i, N_tot, (N_tot - i) / speed))
        time_previous = time.time()

    try:
        tree = il.sublink.loadTree(basePath, n_group, i, fields=fields, onlyMPB=False)
        x = tree['SnapNum']
        Ms_log = log10(tree['SubhaloStellarPhotometricsMassInRad'] * 1e10 / 0.704)
        Mh_log = log10(tree['SubhaloMass'] * 1e10 / 0.704)
        Vmax_log = log10(tree['SubhaloVmax'])
        BH_mass = tree["SubhaloBHMass"] * 1e10 / 0.704


        N_spike = 0

    except:
        # no data available set N_spike=-1
        print("this one fails %d"%i)
        N_spike = -1
        x = -1
        Ms_log = -1
        Mh_log = -1
        Vmax_log = -1
        BH_mass=-1

    hf.create_dataset("SnapNum_zzz".replace("zzz", str(i)), data=x, dtype="int")

    hf.create_dataset("Mh_log_zzz".replace("zzz", str(i)), data=Mh_log, dtype="f")
    hf.create_dataset("Ms_log_zzz".replace("zzz", str(i)), data=Ms_log, dtype="f")
    hf.create_dataset("Vmax_log_zzz".replace("zzz", str(i)), data=Vmax_log, dtype="f")
    hf.create_dataset("BH_mass_zzz".replace("zzz", str(i)), data=BH_mass, dtype="f")

    hf.create_dataset("N_spike_zzz".replace("zzz", str(i)), data=N_spike, dtype="int")
    N_spike_array.append(N_spike)

N_spike_array = np.array(N_spike_array)
hf.create_dataset("N_spike_all", data=N_spike_array, dtype="int")

hf.close()
print("Done")




