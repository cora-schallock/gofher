
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


the_path = "C:\\Users\\school\\Desktop\\cross_id\\sdss_mosaic_construction\\table5\\NGC3169\\NGC3169_color.png"

mo_labels = [['color','color2']]
#mo_labels.extend(get_subplot_mosaic_strtings(bands_in_order))
    
gs_kw = dict(width_ratios=[1,1], height_ratios=[1])
fig, axd = plt.subplot_mosaic(mo_labels,
                              gridspec_kw=gs_kw, figsize = (24,32),
                              constrained_layout=True,num=1, clear=True) #num=1, clear=True #https://stackoverflow.com/a/65910539/13544635

color = mpimg.imread(the_path)
axd['color'].imshow(color)
plt.show()