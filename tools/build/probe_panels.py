import sys, numpy as np
sys.path.insert(0, 'tools/build')
from texlib import load_rgb
from scipy import ndimage
sheet = load_rgb(sys.argv[1]); cols=int(sys.argv[2]); rows=int(sys.argv[3])
H,W,_=sheet.shape
pw, ph = W/cols, H/rows
for r in range(rows):
  for c in range(cols):
    x0,y0=int(c*pw)+5,int(r*ph)+5; x1,y1=int((c+1)*pw)-5,int((r+1)*ph)-5
    crop=sheet[y0:y1,x0:x1]
    border=np.concatenate([crop[0],crop[-1],crop[:,0],crop[:,-1]])
    bg=np.median(border,axis=0)
    d=np.abs(crop-bg).max(2); sat=crop.max(2)-crop.min(2)
    fg=~((d<0.07)&(sat<0.06))
    fg=ndimage.binary_opening(fg,iterations=2); fg=ndimage.binary_fill_holes(fg)
    lab,n=ndimage.label(fg)
    comps=[]
    for i,s in enumerate(ndimage.find_objects(lab)):
      a=(lab[s]==i+1).sum()
      if a<300: continue
      comps.append((s[1].start,s[0].start,s[1].stop,s[0].stop))
    comps.sort(key=lambda b:(b[0]//40, b[1]))
    print(r,c,(x0,y0),len(comps),comps)
