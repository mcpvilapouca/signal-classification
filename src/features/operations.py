import pandas as pd
import numpy as np


def decompact_df(row,column_name,channels):
    for i,channel in enumerate(channels):
        row[channel]=row[column_name][i]
    return row

def ENE(row,col):
    ene=[x**2 for x in row[col]]
    ene=sum(ene)/len(ene)
    return ene

def ENT(row,col):
    log2 = np.log(2)
    val=sum(row[col])
    p=[(x)/val for x in row[col]]

    p1=[]
    for i in p:
        p1.append(i*(i*log2))

    ent=-sum(p1)

    return ent

def ZCC(row,col):
    count=0
    for i in range(len(row[col])-1):
        if row[col][i-1]*row[col][i]<0:
            count=count+1
    return count

def MEAN(row,col):
    mean=sum(row[col])/len(row[col])

    return mean

def STD(row,col):
    std=pd.Series(row[col]).std()

    return std

def MEDIAN(row,col):
    median=pd.Series(row[col]).median()

    return median

def MAXVAL(row,col):
    max=(pd.Series(row[col]).abs()).max()

    return max

def MINVAL(row,col):
    min=pd.Series(row[col]).min()

    return min

def SKEW(row,col):
    skew=pd.Series(row[col]).skew()

    return skew

def KURTOSIS(row,col):
    kurt=pd.Series(row[col]).kurt()

    return kurt

def FFT(row,col):
    FFT=fft(np.array(row[col]))
    FFT=np.real(FFT)
    return FFT

def FFT_imax(row,col):
    i=pd.Series(row[col]).idxmax()
    return i

def get_segments(row,channels,nseg):

    #get data
    for channel in channels:
        #get round number of points per segment
        rseg=int(len(row[channel])/nseg)

        #check how many points are missing since we round reg
        lpoints=len(row[channel])-rseg*nseg

        for i in range(nseg):
            if i==0:
                row[channel+'_'+str(i)]=row[channel][i*rseg:(i*rseg)+rseg]
            elif i==nseg-1:
                row[channel+'_'+str(i)]=row[channel][i*rseg:i*rseg+rseg+lpoints]
            else:
                row[channel+'_'+str(i)]=row[channel][i*rseg+1:(i*rseg)+rseg+1]

    return row