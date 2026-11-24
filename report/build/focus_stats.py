import json, glob, os, numpy as np, cv2
rows=[]
for lp in sorted(glob.glob('labels/pulp/rotor_raw/*_full_labels.json')):
    d=json.load(open(lp)); img=cv2.imread(os.path.join('crops/pulp/rotor_raw', d['image_file']), 0)
    if img is None: continue
    im=cv2.GaussianBlur(img.astype(np.float32),(3,3),0)
    H,W=im.shape
    for b in d['bubbles']:
        if b.get('shape_type')!='circle' or b.get('touches_edge') or b.get('unsure'): continue
        cx,cy,r=b['cx'],b['cy'],b['r']
        if cx-1.6*r<0 or cy-1.6*r<0 or cx+1.6*r>=W or cy+1.6*r>=H: continue
        th=np.linspace(0,2*np.pi,72,endpoint=False); rad=np.arange(0,1.6*r,0.5)
        X=cx+np.outer(np.cos(th),rad); Y=cy+np.outer(np.sin(th),rad)
        P=cv2.remap(im, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR)  # angles x radii
        prof=np.median(P,axis=0)
        g=np.abs(np.gradient(P,0.5,axis=1))
        win=(rad>0.8*r)&(rad<1.2*r)
        sharp=np.median(g[:,win].max(axis=1))
        rim=prof[(rad>0.75*r)&(rad<1.05*r)].min()
        bg=np.median(prof[(rad>1.25*r)&(rad<1.55*r)])
        core=prof[rad<0.4*r].max(); inner=np.median(prof[(rad>0.4*r)&(rad<0.75*r)])
        # edge width: radial distance for 10-90% transition between rim min and bg
        rows.append((sharp, bg-rim, core-inner, rim, bg, 2*r))
a=np.array(rows); print('n',len(a))
for i,n in enumerate(['rim sharpness grey/px','rim darker than surroundings (grey)','core brighter than interior (grey)','rim grey','surround grey','diam px']):
    q=np.percentile(a[:,i],[10,25,50,75,90]); print(f'{n:38s}', ' '.join(f'{v:6.1f}' for v in q))
