import sep
import numpy as np

def calc_sep_size(a,b):
    return 3.14*a*b

def calc_dist(cm_x,cm_y,el_x,el_y):
    return (((cm_x-el_x)*(cm_x-el_x)) + ((cm_y-el_y)*(cm_y-el_y)))**0.5

def run_sep(data, cm_x, cm_y):
    data = data.byteswap().newbyteorder()
    bkg = sep.Background(data)
    data_sub = data - bkg

    #print(bkg.globalback)
    #print(bkg.globalrms)
    mu_bkg = bkg.globalback
    
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
    
    the_el_sep = None
    the_el_sep_dist = np.inf

    for i in objects:
        if calc_sep_size(i['a'],i['b']) < 0.001*data.shape[0]*data.shape[1]: continue
        
        d = calc_dist(cm_x,cm_y,i['x'],i['y'])
        if d < the_el_sep_dist:
            the_el_sep = i; the_el_sep_dist = d
    
    return the_el_sep, mu_bkg