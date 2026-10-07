import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)

BLUE = '#003D7C'
RED = '#E52E5A'
GRAY = '#6B7280'
GREEN = '#2A7F62'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'mathtext.fontset': 'dejavusans',
    'font.family': 'DejaVu Sans',
})

def save(fig, name, folder=OUT):
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight', pad_inches=0.06)
    fig.savefig(folder / f'{name}.png', bbox_inches='tight', pad_inches=0.06, dpi=180)
    plt.close(fig)

# 1. Three notions of error: distance/function/gradient on flat vs steep quadratic
fig, axs = plt.subplots(1,2,figsize=(10.5,3.5))
x=np.linspace(-2.4,2.4,500)
for ax,a,title in [(axs[0],0.25,'Плоская функция'),(axs[1],4.0,'Крутая функция')]:
    y=.5*a*x**2
    ax.plot(x,y,lw=2.4,color=BLUE)
    x0=1.2; y0=.5*a*x0**2; grad=a*x0
    ax.scatter([0,x0],[0,y0],s=[45,55],color=[RED,GREEN],zorder=5)
    ax.vlines(x0,0,y0,colors=GRAY,linestyles=':',lw=1.5)
    ax.annotate(r'$|x-x^\star|$',xy=(x0/2,0),xytext=(x0/2,0.25*max(y.max(),1)),ha='center',arrowprops=dict(arrowstyle='|-|',color=GRAY),fontsize=10)
    ax.annotate(r'$f(x)-f^\star$',xy=(x0,y0/2),xytext=(1.55,y0/2),arrowprops=dict(arrowstyle='->',color=GRAY),fontsize=10)
    ax.text(x0+0.08,y0+0.06*max(y.max(),1),rf'$|f^\prime(x)|={abs(grad):.1f}$',fontsize=10)
    ax.set_title(title); ax.set_xlabel('$x$'); ax.grid(alpha=.15)
axs[0].set_ylabel('$f(x)$')
fig.suptitle('Расстояние до минимума, ошибка по функции и градиент — разные величины',y=1.03)
fig.tight_layout(); save(fig,'error_metrics')

# 2. Rate preview: 1/k vs geometric
fig, axs = plt.subplots(1,2,figsize=(10.4,3.6))
k=np.arange(1,101)
axs[0].plot(k,1/k,lw=2.3,color=BLUE,label=r'$1/k$')
axs[0].plot(k,0.9**k,lw=2.3,color=RED,label=r'$0.9^k$')
axs[0].plot(k,0.99**k,lw=2.0,color=GREEN,label=r'$0.99^k$')
axs[0].set_xlabel('$k$'); axs[0].set_ylabel('ошибка (нормированная)'); axs[0].set_title('Обычный масштаб'); axs[0].legend(); axs[0].grid(alpha=.15)
axs[1].semilogy(k,1/k,lw=2.3,color=BLUE,label=r'$1/k$')
axs[1].semilogy(k,0.9**k,lw=2.3,color=RED,label=r'$0.9^k$')
axs[1].semilogy(k,0.99**k,lw=2.0,color=GREEN,label=r'$0.99^k$')
axs[1].set_xlabel('$k$'); axs[1].set_title('Логарифмическая шкала'); axs[1].legend(); axs[1].grid(alpha=.15,which='both')
fig.suptitle('Два режима: субгеометрическое и геометрическое уменьшение ошибки',y=1.03)
fig.tight_layout(); save(fig,'rate_preview')

# 3. Telescoping visualization
fig, ax = plt.subplots(figsize=(10.2,2.8))
ax.axis('off')
ys=[0.72,0.47,0.22]
labels=[r'$R_0^2-R_1^2$', r'$R_1^2-R_2^2$', r'$R_2^2-R_3^2$']
for i,(y,lbl) in enumerate(zip(ys,labels)):
    ax.text(0.05,y,lbl,fontsize=20,color=BLUE)
    if i<2:
        ax.text(0.38,y,r'$+\;$',fontsize=20,color=GRAY)
ax.text(0.49,0.47,r'$\cdots$',fontsize=24,color=GRAY)
ax.text(0.64,0.47,r'$+\;(R_{N-1}^2-R_N^2)$',fontsize=20,color=BLUE)
ax.text(0.12,0.02,r'$=\;R_0^2-R_N^2\;\leq\;R_0^2$',fontsize=24,color=RED)
ax.set_title('Телескопическая сумма: внутренние расстояния сокращаются',fontsize=13)
save(fig,'telescoping_sum')

# 4. 1/k meaning and epsilon target
fig, ax = plt.subplots(figsize=(7.6,4.1))
k=np.arange(1,101)
c=1.0
ax.plot(k,c/k,lw=2.4,color=BLUE)
for eps in [0.2,0.1,0.05]:
    kk=int(np.ceil(c/eps))
    ax.axhline(eps,lw=1.0,ls='--',color=GRAY,alpha=.8)
    ax.scatter([kk],[c/kk],s=45,color=RED,zorder=5)
    ax.text(kk+2,eps+0.006,rf'$\varepsilon={eps}$: $k\approx{kk}$',fontsize=9)
ax.set_xlabel('$k$'); ax.set_ylabel(r'$C/k$'); ax.set_title(r'Чтобы уменьшить $C/k$ в 10 раз, нужно примерно в 10 раз больше итераций')
ax.grid(alpha=.15); fig.tight_layout(); save(fig,'one_over_k')

# 5. Strong convex lower support
fig, ax = plt.subplots(figsize=(7.8,4.2))
x=np.linspace(-2.4,2.4,500)
f=.5*x**2+.05*x**4
x0=-0.9; f0=.5*x0**2+.05*x0**4; g=x0+.2*x0**3; mu=1.0
lin=f0+g*(x-x0); quad=lin+.5*mu*(x-x0)**2
ax.plot(x,f,lw=2.4,color=BLUE,label='$f(y)$')
ax.plot(x,lin,lw=1.6,ls='--',color=GRAY,label='касательная')
ax.plot(x,quad,lw=2.2,color=RED,label=r'касательная $+\frac{\mu}{2}(y-x)^2$')
ax.scatter([x0],[f0],s=50,color=RED,zorder=5)
ax.fill_between(x,quad,f,where=f>=quad,color=BLUE,alpha=.06)
ax.set_xlabel('$y$'); ax.set_ylabel('значение'); ax.set_title('Сильная выпуклость даёт квадратичную нижнюю опору')
ax.legend(fontsize=8.5); ax.grid(alpha=.15); fig.tight_layout(); save(fig,'strong_lower_model')

# 6. Gradient gap relation on quadratics
fig, axs=plt.subplots(1,2,figsize=(10.4,3.6))
x=np.linspace(-2.2,2.2,500)
for ax,a,title in [(axs[0],1.0,r'$\mu=1$'),(axs[1],4.0,r'$\mu=4$')]:
    f=.5*a*x**2; g=a*x
    ax.plot(x,f,lw=2.3,color=BLUE,label=r'$f(x)-f^\star$')
    ax.plot(x,g**2/(2*a),lw=2.0,ls='--',color=RED,label=r'$\|\nabla f(x)\|^2/(2\mu)$')
    ax.set_title(title); ax.set_xlabel('$x$'); ax.grid(alpha=.15)
axs[0].set_ylabel('величина'); axs[0].legend(fontsize=8.5); axs[1].legend(fontsize=8.5)
fig.suptitle(r'Для квадратичной функции неравенство $\|\nabla f\|^2\geq2\mu(f-f^\star)$ достигается с равенством',y=1.03)
fig.tight_layout(); save(fig,'gradient_gap_relation')

# 7. Kappa contraction curves
fig, ax = plt.subplots(figsize=(7.8,4.2))
k=np.arange(0,151)
for kap,col in [(2,RED),(10,BLUE),(100,GREEN)]:
    q=1-1/kap
    ax.plot(k,q**k,lw=2.3,color=col,label=rf'$\kappa={kap},\;q={q:.2f}$')
ax.set_yscale('log'); ax.set_xlabel('$k$'); ax.set_ylabel('верхняя оценка относительной ошибки'); ax.set_title('Большая обусловленность замедляет геометрическую сходимость')
ax.legend(); ax.grid(alpha=.15,which='both'); fig.tight_layout(); save(fig,'kappa_rates')

def make_rotated_A(kappa, angle=0.55):
    c,s=np.cos(angle),np.sin(angle)
    Q=np.array([[c,-s],[s,c]])
    D=np.diag([1.0,float(kappa)])
    return Q@D@Q.T

def gd(A,x0,n=80):
    L=np.linalg.eigvalsh(A).max(); alpha=1.0/L
    xs=[x0.copy()]
    x=x0.copy()
    for _ in range(n):
        x=x-alpha*(A@x)
        xs.append(x.copy())
    return np.array(xs),alpha

# 8. Trajectories for kappa 1,10,100
fig, axs=plt.subplots(1,3,figsize=(11.7,3.6))
for ax,kap in zip(axs,[1,10,100]):
    A=make_rotated_A(kap)
    xx=np.linspace(-2.6,2.6,260); yy=np.linspace(-2.2,2.2,240); X,Y=np.meshgrid(xx,yy)
    Z=.5*(A[0,0]*X**2+2*A[0,1]*X*Y+A[1,1]*Y**2)
    lev=np.geomspace(0.08,20,9)
    ax.contour(X,Y,Z,levels=lev,linewidths=0.9,colors=BLUE,alpha=.65)
    path,_=gd(A,np.array([2.2,1.7]),n=55)
    ax.plot(path[:,0],path[:,1],'-o',ms=2.3,lw=1.5,color=RED)
    ax.scatter([0],[0],s=40,color=GREEN,zorder=5)
    ax.set_title(rf'$\kappa={kap}$'); ax.set_aspect('equal'); ax.set_xlim(-2.6,2.6); ax.set_ylim(-2.2,2.2); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(r'Один и тот же GD с шагом $1/L$: вытянутая геометрия приводит к медленному движению',y=1.03)
fig.tight_layout(); save(fig,'gd_trajectories_kappa')

# 9. Objective gaps + theoretical bounds
fig, ax=plt.subplots(figsize=(7.8,4.2))
for kap,col in [(2,RED),(10,BLUE),(100,GREEN)]:
    A=make_rotated_A(kap,angle=.4)
    xs,_=gd(A,np.array([2.0,1.6]),n=120)
    vals=np.array([.5*x@(A@x) for x in xs])
    norm=vals/vals[0]
    q=1-1/kap
    ax.semilogy(np.arange(len(norm)),norm,lw=2.2,color=col,label=rf'эксперимент, $\kappa={kap}$')
    ax.semilogy(np.arange(len(norm)),q**np.arange(len(norm)),ls='--',lw=1.2,color=col,alpha=.8)
ax.set_xlabel('$k$'); ax.set_ylabel(r'$(f(x_k)-f^\star)/(f(x_0)-f^\star)$'); ax.set_title('Фактическая ошибка и теоретическая геометрическая граница')
ax.grid(alpha=.15,which='both'); ax.legend(fontsize=8.5); fig.tight_layout(); save(fig,'objective_gaps')

# 10. Theory bound vs actual for one case
fig, ax=plt.subplots(figsize=(7.8,4.2))
kap=20; A=make_rotated_A(kap,angle=.7); xs,_=gd(A,np.array([2.2,-1.5]),n=100)
vals=np.array([.5*x@(A@x) for x in xs]); rel=vals/vals[0]; q=1-1/kap; bound=q**np.arange(len(rel))
ax.semilogy(rel,lw=2.4,color=BLUE,label='фактическая ошибка')
ax.semilogy(bound,lw=2.0,ls='--',color=RED,label='гарантированная верхняя оценка')
ax.fill_between(np.arange(len(rel)),rel,bound,where=(bound>=rel),color=BLUE,alpha=.06)
ax.set_xlabel('$k$'); ax.set_ylabel('относительная ошибка'); ax.set_title(r'Теорема гарантирует «не хуже чем», а не точную траекторию')
ax.legend(); ax.grid(alpha=.15,which='both'); fig.tight_layout(); save(fig,'theory_vs_experiment')

# 11. Epsilon complexity comparison
fig, ax=plt.subplots(figsize=(8.0,4.2))
eps=np.logspace(-4,-1,200)
conv=1/eps
strong=np.log(1/eps)
conv=conv/conv[0]; strong=strong/strong[0]
ax.loglog(eps,conv,lw=2.3,color=BLUE,label=r'выпуклая: $O(1/\varepsilon)$')
ax.loglog(eps,strong,lw=2.3,color=RED,label=r'сильно выпуклая: $O(\log(1/\varepsilon))$')
ax.invert_xaxis(); ax.set_xlabel(r'требуемая точность $\varepsilon$'); ax.set_ylabel('относительное число итераций (схематично)'); ax.set_title('Ужесточение точности по-разному влияет на сложность')
ax.legend(); ax.grid(alpha=.15,which='both'); fig.tight_layout(); save(fig,'complexity_epsilon')

print('generated', len(list(OUT.glob('*.pdf'))), 'lecture figures')
