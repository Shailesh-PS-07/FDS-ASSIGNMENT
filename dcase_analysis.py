import glob,os,hashlib,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.metrics import roc_auc_score,roc_curve
R='data/dev_data_fan_small'; NPS=16384; SR=16000
def load(f): sr,x=wavfile.read(f); return x.astype(float)/32768
tr=sorted(glob.glob(R+'/train/*.wav')); te=sorted(glob.glob(R+'/test/*.wav'))
y=np.array([int('anomaly' in f) for f in te]); X={f:load(f) for f in tr+te}
fr=np.fft.rfftfreq(NPS,1/SR)
def rmsdb(x): return 20*np.log10(np.sqrt(((x-x.mean())**2).mean()))
def logspec(x):                       # energy-normalised mean log power spectrum
    x=x-x.mean(); x=x/np.sqrt((x**2).mean())
    _,_,Z=stft(x,SR,nperseg=NPS,noverlap=NPS//2); return np.log((np.abs(Z)**2).mean(1)+1e-12)
Ltr=np.array([logspec(X[f]) for f in tr]); Lte=np.array([logspec(X[f]) for f in te])
mu,sd=Ltr.mean(0),Ltr.std(0)+1e-9
def score(L): return ((mu-L)/sd).max(1)          # spectral-line deficit score (one-sided)
s=score(Lte); rms=np.array([rmsdb(X[f]) for f in te])
# leave-one-out train scores -> threshold from normals only
loo=[]
for i in range(len(tr)):
    m=np.delete(Ltr,i,0); loo.append((((m.mean(0)-Ltr[i])/(m.std(0)+1e-9)).max()))
thr=np.percentile(loo,95)
pred=s>thr
print('AUC spectral %.3f | AUC RMS %.3f | AUC |RMS-median| %.3f'%(roc_auc_score(y,s),roc_auc_score(y,rms),roc_auc_score(y,np.abs(rms-np.median(rms[y==0])))))
print('threshold (95th pct LOO train) %.2f  TPR %.3f  FPR %.3f  train LOO FPR %.3f'%(thr,pred[y==1].mean(),pred[y==0].mean(),(np.array(loo)>thr).mean()))
# resolution robustness
for n in [8192,12288,16384,24576,32768]:
    def ls(x,n=n):
        x=x-x.mean();x=x/np.sqrt((x**2).mean());_,_,Z=stft(x,SR,nperseg=n,noverlap=n//2);return np.log((np.abs(Z)**2).mean(1)+1e-12)
    a=np.array([ls(X[f]) for f in tr]);b=np.array([ls(X[f]) for f in te])
    print('nperseg',n,'AUC %.3f'%roc_auc_score(y,((a.mean(0)-b)/(a.std(0)+1e-9)).max(1)))
# gain invariance: random gain -30..+30 dB on every test file
rng=np.random.default_rng(0); gains=np.linspace(0,30,7); auc_s=[];auc_r=[]
for G in gains:
    g=10**(rng.uniform(-G,G,len(te))/20)
    Lg=np.array([logspec(X[f]*gi) for f,gi in zip(te,g)]); auc_s.append(roc_auc_score(y,score(Lg)))
    auc_r.append(roc_auc_score(y,[rmsdb(X[f]*gi) for f,gi in zip(te,g)]))
print('gain sweep spectral',np.round(auc_s,3),'rms',np.round(auc_r,3))
# quiet-only
medN=np.median(rms[y==0]); q=(rms<=medN)
qa=(y==1)&q; print('anomalies quieter than normal median:',qa.sum(),'AUC vs normals %.3f'%roc_auc_score(np.r_[np.zeros((y==0).sum()),np.ones(qa.sum())],np.r_[s[y==0],s[qa]]),' TPR@thr %.3f'%(s[qa]>thr).mean())
lo=(rms<np.percentile(rms,50)); print('lower-energy half of test: n=%d anom=%d AUC spectral %.3f AUC rms %.3f'%(lo.sum(),y[lo].sum(),roc_auc_score(y[lo],s[lo]),roc_auc_score(y[lo],rms[lo])))
# case study: quietest anomaly
names=[os.path.basename(f) for f in te]
ai=np.where(y==1)[0]; ci=ai[np.argmin(rms[ai])]
loud_rank=(-rms).argsort().argsort()+1; sc_rank=(-s).argsort().argsort()+1
print('CASE',names[ci],'rms %.2f dB loudness rank %d/%d, spectral rank %d/%d score %.2f'%(rms[ci],loud_rank[ci],len(te),sc_rank[ci],len(te),s[ci]))
print('quietest file in whole test set is anomaly:',y[np.argmin(rms)]==1)
# harmonics for case study vs normal median: fundamental & 2nd/3rd harmonic ratios
def peak(L,f0,w=3):
    b=(fr>=f0-w)&(fr<=f0+w); i=np.argmax(L[b]); return fr[b][i],L[b][i]
def harm(L):
    f1,p1=peak(L,67,6); out=[f1]
    for k in (2,3,4): fk,pk=peak(L,k*f1,4); out.append(pk-p1)
    return out
print('case harmonics (f0, dB rel: h2,h3,h4):',np.round(harm(Lte[ci]),2))
nrm=np.array([harm(Lte[i]) for i in np.where(y==0)[0]]); print('normal mean',np.round(nrm.mean(0),2),'std',np.round(nrm.std(0),2))
# ---------- FIGURE ----------
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(18,11.5)); gs=fig.add_gridspec(2,3,hspace=0.5,wspace=0.32)
CN,CA,CQ='#2a6fdb','#d64545','#f0a020'
# 1 loudness overlap
ax=fig.add_subplot(gs[0,0]); bins=np.linspace(rms.min()-0.1,rms.max()+0.1,28)
ax.hist(rms[y==0],bins,color=CN,alpha=.75,label='Normal (n=23)'); ax.hist(rms[y==1],bins,color=CA,alpha=.6,label='Anomaly (n=28)')
ax.axvline(medN,color='k',ls='--',lw=1); ax.text(medN+.03,ax.get_ylim()[1]*.9,'normal median',fontsize=8)
ax.set_xlabel('RMS level (dB re full scale)'); ax.set_ylabel('files'); ax.legend(frameon=False)
ax.set_title('① Loudness overlaps heavily\n%d/28 anomalies quieter than normal median'%qa.sum(),loc='left',fontsize=11,fontweight='bold')
# 2 ROC
ax=fig.add_subplot(gs[0,1])
for lab,sc_,c,ls in [('Spectral-line deficit (energy-normalised)',s,CA,'-'),('RMS "decibel meter"',rms,'#777','--')]:
    fpr,tpr,_=roc_curve(y,sc_); ax.plot(fpr,tpr,c=c,ls=ls,lw=2.2,label='%s  AUC=%.2f'%(lab,roc_auc_score(y,sc_)))
fpr,tpr,_=roc_curve(np.r_[np.zeros((y==0).sum()),np.ones(qa.sum())],np.r_[s[y==0],s[qa]]); ax.plot(fpr,tpr,c=CQ,lw=2.2,label='Spectral, quiet anomalies only  AUC=%.2f'%roc_auc_score(np.r_[np.zeros((y==0).sum()),np.ones(qa.sum())],np.r_[s[y==0],s[qa]]))
ax.plot([0,1],[0,1],':',c='#bbb'); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.legend(frameon=False,fontsize=8,loc='lower right')
ax.set_title('② ROC: spectral vs loudness meter\n(energy-independent spectral AUC)',loc='left',fontsize=11,fontweight='bold')
# 3 scatter
ax=fig.add_subplot(gs[0,2])
ax.scatter(rms[y==0],s[y==0],c=CN,s=45,label='Normal',edgecolor='w')
ax.scatter(rms[(y==1)&~q],s[(y==1)&~q],c=CA,s=45,label='Anomaly (louder)',edgecolor='w')
ax.scatter(rms[qa],s[qa],c=CQ,s=70,marker='D',label='Anomaly (quiet, ≤ normal median)',edgecolor='k',lw=.6)
ax.scatter(rms[ci],s[ci],s=260,facecolors='none',edgecolors='k',lw=2); ax.annotate('quietest file overall\n(an anomaly)',(rms[ci],s[ci]),xytext=(-39.2,s[ci]+3.5),fontsize=8,arrowprops=dict(arrowstyle='->'))
ax.axhline(thr,c='k',ls='--',lw=1); ax.text(rms.max(),thr+.15,'threshold (from train only)',ha='right',fontsize=8)
ax.set_xlabel('RMS level (dB)'); ax.set_ylabel('Spectral anomaly score'); ax.legend(frameon=False,fontsize=8,loc='upper left')
ax.set_title('③ Score vs loudness: quiet anomalies\nstill sit far above the threshold',loc='left',fontsize=11,fontweight='bold')
# 4 case study spectrum
ax=fig.add_subplot(gs[1,0:2]); b=(fr>=20)&(fr<=420)
lo_,hi_=mu-2*sd,mu+2*sd
ax.fill_between(fr[b],lo_[b],hi_[b],color=CN,alpha=.18,label='Normal training range (±2σ, 63 files)')
ax.plot(fr[b],mu[b],c=CN,lw=1.2,label='Normal mean')
ni=np.where(y==0)[0][0]; ax.plot(fr[b],Lte[ni][b],c=CN,lw=.8,alpha=.6,ls=':',label='A test normal file')
ax.plot(fr[b],Lte[ci][b],c=CA,lw=1.4,label='Case study: %s (%.2f dB, quietest of all %d files)'%(names[ci],rms[ci],len(te)))
for k in range(1,6): ax.axvline(67.4*k,c='#bbb',lw=.7,ls='--')
ax.set_ylim(-11,-1.5); ax.text(67.4,-1.7,' f0≈67 Hz',fontsize=8,va='top'); ax.text(134.8,-1.7,' 2f0',fontsize=8,va='top'); ax.text(202,-1.7,' 3f0',fontsize=8,va='top')
ax.annotate('missing f0 line\n(z = 9.6)',(67.4,Lte[ci][np.argmin(abs(fr-67.4))]),xytext=(85,-3.0),fontsize=8,arrowprops=dict(arrowstyle='->',color=CA),color=CA); ax.annotate('extra off-harmonic\nline ~122 Hz',(122.1,Lte[ci][np.argmin(abs(fr-122.1))]),xytext=(150,-3.2),fontsize=8,arrowprops=dict(arrowstyle='->',color=CA),color=CA)
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Log power (energy-normalised)'); ax.legend(frameon=False,fontsize=8,loc='lower right',ncol=2)
ax.set_title('④ Case study: quietest recording of all 51 is an anomaly – f0 (67 Hz) & 2f0 lines vanish, off-harmonic line at ~122 Hz',loc='left',fontsize=11,fontweight='bold')
# 5 gain invariance + data quality text
ax=fig.add_subplot(gs[1,2]); ax.plot(gains,auc_s,'o-',c=CA,lw=2,label='Spectral (energy-normalised)'); ax.plot(gains,auc_r,'s--',c='#777',lw=2,label='RMS meter')
ax.axhline(.5,c='#bbb',ls=':'); ax.set_ylim(0,1.05); ax.set_xlabel('Random gain applied to each test file (± dB)'); ax.set_ylabel('AUC'); ax.legend(frameon=False,fontsize=8,loc='center left')
ax.set_title('⑤ Random gain stress test:\nspectral AUC unchanged, RMS ≈ chance',loc='left',fontsize=11,fontweight='bold')
fig.suptitle('DCASE fan anomaly detection – energy-independent spectral analysis  |  spectral AUC %.2f vs RMS-meter AUC %.2f   |  TPR %.0f%% @ FPR %.0f%%'%(roc_auc_score(y,s),roc_auc_score(y,rms),100*pred[y==1].mean(),100*pred[y==0].mean()),fontsize=13,fontweight='bold',y=.99)
fig.savefig('/mnt/user-data/outputs/dcase_energy_independent_analysis.png',dpi=140,bbox_inches='tight')
