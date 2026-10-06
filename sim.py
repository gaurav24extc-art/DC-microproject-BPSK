import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc
rng=np.random.default_rng(7)
sps=8; Eb=1.0
pulse=np.ones(sps)/np.sqrt(sps)   # unit-energy rectangular pulse
mf=pulse[::-1]
ebn0_db=np.arange(0,11)
Nbits=1_000_000
th=[];sim=[];errs=[]
for db in ebn0_db:
    n0=Eb/10**(db/10)
    bits=rng.integers(0,2,Nbits)
    sym=2*bits-1
    tx=np.repeat(sym,sps)*np.tile(pulse,Nbits)
    noise=rng.normal(0,np.sqrt(n0/2),tx.size)
    rx=tx+noise
    y=np.convolve(rx,mf)[sps-1::sps][:Nbits]
    det=(y>0).astype(int)
    e=int(np.sum(det!=bits))
    errs.append(e);sim.append(e/Nbits)
    th.append(0.5*erfc(np.sqrt(10**(db/10))))
res={"ebn0":ebn0_db.tolist(),"sim":sim,"th":th,"errs":errs,"N":Nbits}
json.dump(res,open("res.json","w"))
# Fig1 BER
plt.figure(figsize=(6.5,4.5))
plt.semilogy(ebn0_db,th,'b-o',label='Theoretical  Q(√(2Eb/N0))')
plt.semilogy(ebn0_db,sim,'rx--',ms=8,label='Simulated (matched filter)')
plt.grid(True,which='both',alpha=.4);plt.xlabel('Eb/N0 (dB)');plt.ylabel('Bit Error Rate')
plt.title('BPSK BER over AWGN');plt.legend();plt.ylim(1e-6,1)
plt.tight_layout();plt.savefig('ber.png',dpi=150);plt.close()
# Fig2 waveforms at 4 dB, 10 bits
nb=10;db=4;n0=Eb/10**(db/10)
bits=rng.integers(0,2,nb);sym=2*bits-1
tx=np.repeat(sym,sps)*np.tile(pulse,nb)
rx=tx+rng.normal(0,np.sqrt(n0/2),tx.size)
yfull=np.convolve(rx,mf)
y=yfull[:nb*sps]
fig,ax=plt.subplots(3,1,figsize=(7,6.5),sharex=True)
t=np.arange(nb*sps)/sps
ax[0].step(t,tx,where='post');ax[0].set_title('Transmitted BPSK waveform  bits='+''.join(map(str,bits)));ax[0].grid(alpha=.4)
ax[1].plot(t,rx,'g');ax[1].set_title('Received signal (Eb/N0 = 4 dB)');ax[1].grid(alpha=.4)
ax[2].plot(t,y,'r');ax[2].stem(np.arange(1,nb+1),yfull[sps-1::sps][:nb],linefmt='k-',markerfmt='ko',basefmt=' ')
ax[2].set_title('Matched filter output with sampling instants');ax[2].set_xlabel('Time (bit periods)');ax[2].grid(alpha=.4)
plt.tight_layout();plt.savefig('wave.png',dpi=150);plt.close()
# Fig3 scatter of MF outputs at 4 dB and 10 dB
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
for a,db in zip(ax,[4,10]):
    n0=Eb/10**(db/10);N=2000
    b=rng.integers(0,2,N);s=2*b-1
    tx=np.repeat(s,sps)*np.tile(pulse,N);rx=tx+rng.normal(0,np.sqrt(n0/2),tx.size)
    y=np.convolve(rx,mf)[sps-1::sps][:N]
    a.scatter(y,rng.normal(0,.05,N),s=3,c=np.where(y>0,'b','r'));a.axvline(0,c='k',ls='--')
    a.set_title(f'MF samples, Eb/N0={db} dB');a.set_yticks([]);a.set_xlabel('In-phase');a.grid(alpha=.3)
plt.tight_layout();plt.savefig('const.png',dpi=150);plt.close()
for d,s_,t_,e in zip(ebn0_db,sim,th,errs):print(d,e,f"{s_:.2e}",f"{t_:.2e}")
