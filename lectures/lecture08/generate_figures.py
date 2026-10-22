"""Illustrations for lecture 8. All figures are generated with Python/matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa

OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
BLUE='#003D7C'; RED='#DE3055'; GREEN='#24745C'; GRAY='#667085'; LIGHT='#E8EFF7'; GOLD='#D79A2B'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':12,
                     'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.10)
    plt.close(fig)
def clean(ax):
    ax.grid(alpha=.14); ax.set_axisbelow(True)

def quad(x):return .5*x*x
# 1d step regimes: from x=2 on f=x², small/good/too large
fig,axs=plt.subplots(1,3,figsize=(12,3.35),sharex=True,sharey=True)
xx=np.linspace(-3.2,3.2,450)
for ax,a,t in zip(axs,[.08,.45,1.15],['Маленький шаг','Подходящий шаг','Слишком большой шаг']):
    ax.plot(xx,xx**2,color=BLUE,lw=2.4)
    xs=[2.];
    for k in range(4):xs.append(xs[-1]-a*2*xs[-1])
    good=[z for z in xs if abs(z)<=3.2]
    ax.scatter(good,np.array(good)**2,color=RED,s=37,zorder=4)
    for z,w in zip(xs[:-1],xs[1:]):
        if abs(z)<=3.2 and abs(w)<=3.2:
            ax.annotate('',xy=(w,w*w),xytext=(z,z*z),arrowprops={'arrowstyle':'->','color':RED,'lw':1.4})
    ax.set_title(t+f' ($\\alpha={a:g}$)'); ax.set_xlabel('$x$');ax.set_xlim(-3.2,3.2);ax.set_ylim(0,11);clean(ax)
axs[0].set_ylabel('$f(x)=x^2$');fig.tight_layout();save(fig,'step_1d')

# 2d contours with trajectories 3 alphas
A=np.diag([1.,18.]);X,Y=np.meshgrid(np.linspace(-3,3,301),np.linspace(-2.1,2.1,240));Z=.5*(X**2+18*Y**2)
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharex=True,sharey=True)
for ax,alpha,title in zip(axs,[.012,.095,.12],['Медленно','Устойчиво, с колебаниями','Расходится']):
    ax.contour(X,Y,Z,levels=[.25,.75,1.5,3,6,12,20,30],colors=BLUE,alpha=.45,linewidths=.95)
    p=np.array([2.8,1.55]); ps=[p.copy()]
    for i in range(13):p=(np.eye(2)-alpha*A)@p;ps.append(p.copy())
    ps=np.array(ps);ax.plot(ps[:,0],ps[:,1],'-o',color=RED,lw=1.45,markersize=2.8)
    ax.scatter([0],[0],marker='x',s=42,color=GREEN,zorder=4)
    ax.set_title(title+f' ($\\alpha={alpha:.3f}$)',fontsize=11);ax.set_xlim(-3,3);ax.set_ylim(-2.1,2.1);ax.set_xlabel('$x_1$');clean(ax)
axs[0].set_ylabel('$x_2$');fig.tight_layout();save(fig,'step_contours')

# 3 derivative threshold values for positive descent
fig,ax=plt.subplots(figsize=(8.4,3.15))
x=np.linspace(0,.18,400); mu,L=1,18
ax.plot(x,np.abs(1-L*x),lw=2.4,color=RED,label=r'$|1-\alpha L|$')
ax.plot(x,np.abs(1-mu*x),lw=2,color=BLUE,label=r'$|1-\alpha\mu|$')
ax.axhline(1,color=GRAY,ls='--',lw=1);ax.axvline(2/L,color=GRAY,ls=':',lw=1.6)
ax.fill_between(x,0,1,where=(x<2/L),color=GREEN,alpha=.07)
ax.text(.026,.16,'область сходимости',color=GREEN,fontsize=11)
ax.set_xlim(0,.18);ax.set_ylim(0,2.3);ax.set_xlabel(r'шаг $\alpha$');ax.set_ylabel('модуль множителя ошибки');ax.legend(loc='upper left',fontsize=10);clean(ax)
fig.tight_layout();save(fig,'spectral_threshold')

# geometric line on bowl + 1d profile
fig=plt.figure(figsize=(11,4.1));ax=fig.add_subplot(121,projection='3d')
x=np.linspace(-1.8,2.4,90);y=np.linspace(-1.7,2.0,85);xx,yy=np.meshgrid(x,y);zz=.5*((xx-.25)**2+3*(yy+.3)**2)
ax.plot_surface(xx,yy,zz,cmap='Blues',alpha=.72,linewidth=0)
p=np.array([1.55,1.20]);d=np.array([-.65,-.76]);a=np.linspace(0,2.2,100);ps=p[None,:]+a[:,None]*d
f=.5*((ps[:,0]-.25)**2+3*(ps[:,1]+.3)**2)
ax.plot(ps[:,0],ps[:,1],f+0.015,color=RED,lw=3)
ax.scatter([p[0]],[p[1]],[.5*((p[0]-.25)**2+3*(p[1]+.3)**2)],s=42,c=RED)
ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.set_zlabel('$f$');ax.view_init(elev=28,azim=-59);ax.set_title('Путь на поверхности',pad=4)
ax2=fig.add_subplot(122);ax2.plot(a,f,lw=2.6,color=BLUE);ax2.axvline(a[np.argmin(f)],ls=':',color=RED,lw=1.5);ax2.scatter([a[np.argmin(f)]],[min(f)],s=46,color=RED)
ax2.set_xlabel(r'расстояние по направлению $\alpha$');ax2.set_ylabel(r'$\varphi(\alpha)$');ax2.set_title('Значения вдоль направления');clean(ax2)
fig.tight_layout();save(fig,'line_section_3d')

# directional slope plot lines at initial point
fig,ax=plt.subplots(figsize=(7.6,3.8));t=np.linspace(0,1.7,450)
for val,c,style,lab in [(-2.,BLUE,'-','Направление убывания'),(0.,GRAY,'--','Касательное направление'),(1.4,RED,'-','Направление роста')]:
    y=1+val*t+.85*t*t
    ax.plot(t,y,color=c,ls=style,lw=2.3,label=lab)
ax.scatter([0],[1],s=45,color='black',zorder=3);ax.set_xlim(0,1.7);ax.set_xlabel(r'шаг $\alpha$');ax.set_ylabel(r'$\varphi(\alpha)$');ax.legend(fontsize=9,loc='upper center');clean(ax)
fig.tight_layout();save(fig,'directional_slope')

# quadratic exact alpha line profile
fig,ax=plt.subplots(figsize=(7.8,3.9))
g=np.array([3.,2.]);AA=np.diag([2.,9.]);aa=np.linspace(0,.35,400)
ff=-aa*np.dot(g,g)+.5*aa*aa*np.dot(g,AA@g)
star=np.dot(g,g)/np.dot(g,AA@g)
ax.plot(aa,ff,lw=2.6,color=BLUE);ax.axvline(star,color=RED,ls='--',lw=1.5)
ax.scatter([star],[-star*np.dot(g,g)+.5*star**2*np.dot(g,AA@g)],color=RED,s=58,zorder=5)
ax.annotate(r'$\alpha^*=\|g\|^2/(g^TAg)$',xy=(star,np.min(ff)),xytext=(.17,-.45),arrowprops={'arrowstyle':'->','color':RED},fontsize=12)
ax.axhline(0,lw=.7,color=GRAY);ax.set_xlabel(r'$\alpha$');ax.set_ylabel(r'$f(x-\alpha g)-f(x)$');ax.set_xlim(0,.35);clean(ax)
fig.tight_layout();save(fig,'exact_alpha_profile')

# exact line search orthogonal consecutive gradients example
fig,ax=plt.subplots(figsize=(7.8,4.15))
AA=np.diag([1.,12.]); xx,yy=np.meshgrid(np.linspace(-2.6,2.6,300),np.linspace(-1.4,1.4,300)); zz=.5*(xx*xx+12*yy*yy)
ax.contour(xx,yy,zz,levels=[.15,.45,1,2,4,7,11],colors=BLUE,alpha=.42)
p=np.array([2.,.9]); ps=[p.copy()]
for i in range(5):
    g=AA@p;alpha=(g@g)/(g@(AA@g));p=p-alpha*g;ps.append(p.copy())
ps=np.array(ps);ax.plot(ps[:,0],ps[:,1],'-o',color=RED,lw=2,markersize=5);ax.scatter([0],[0],marker='x',c=GREEN,s=60)
ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.set_aspect('equal');ax.set_xlim(-2.6,2.6);ax.set_ylim(-1.4,1.4);ax.set_title('Точный поиск по антиградиенту: следующие направления ортогональны',fontsize=11);clean(ax)
fig.tight_layout();save(fig,'exact_path')

# ideal step vs merely descent
fig,ax=plt.subplots(figsize=(8,3.9));x=np.linspace(0,2,450);y=(x-1)**2+.35
ax.plot(x,y,color=BLUE,lw=2.5);ax.axhline(y[0],color=GRAY,ls='--',lw=1.4,label='исходное значение')
ax.fill_between(x,y,y[0],where=(y<y[0]),alpha=.10,color=GREEN)
ax.scatter([1],[.35],s=55,c=RED);ax.text(1.06,.44,'точный минимум',color=RED,fontsize=11)
ax.set_xlabel(r'$\alpha$');ax.set_ylabel(r'$\varphi(\alpha)$');ax.set_ylim(.15,1.75);ax.set_xlim(0,2);ax.set_title('Многие шаги уменьшают функцию; лишь один даёт точный минимум');clean(ax)
fig.tight_layout();save(fig,'acceptable_steps')

# armijo main chart alpha horizontal: parabola vs threshold line
fig,ax=plt.subplots(figsize=(9,4.2));a=np.linspace(0,1.45,500);y=4-8*a+9*a*a
c=.25; threshold=4-c*8*a
ax.plot(a,y,color=BLUE,lw=2.6,label=r'истинное $\varphi(\alpha)$')
ax.plot(a,4-8*a,color=GRAY,ls=':',lw=1.6,label='линейное приближение')
ax.plot(a,threshold,color=RED,ls='--',lw=2.2,label='граница Армихо')
mask=y<=threshold
ax.fill_between(a,np.minimum(y,threshold),np.maximum(y,threshold),where=mask,color=GREEN,alpha=.12)
maxa=(8*(1-c))/9
ax.axvline(maxa,color=GREEN,lw=1.5,ls=':');ax.text(maxa+.018,1.0,r'$\alpha_{max}$',color=GREEN,fontsize=12)
ax.scatter([0],[4],s=48,color='black',zorder=5)
ax.set_xlim(0,1.45);ax.set_ylim(-1,9);ax.set_xlabel(r'шаг $\alpha$');ax.set_ylabel(r'$\varphi(\alpha)$');ax.legend(fontsize=10,loc='upper left');ax.set_title('Армихо: истинная функция должна быть ниже красной границы');clean(ax)
fig.tight_layout();save(fig,'armijo_geometry')

# c parameter influence
fig,ax=plt.subplots(figsize=(8,3.85));a=np.linspace(0,1.35,500);v=4-8*a+9*a*a
ax.plot(a,v,color=BLUE,lw=2.5,label='значение функции')
for c,col in [(.1,GREEN),(.5,GOLD),(.9,RED)]:
    ax.plot(a,4-c*8*a,lw=2,ls='--',color=col,label=rf'$c={c:g}$')
ax.set_ylim(-.5,8);ax.set_xlabel(r'$\alpha$');ax.set_ylabel(r'$\varphi(\alpha)$');ax.set_title('Большее $c$: более строгое требование уменьшения');ax.legend(fontsize=10,loc='upper left');clean(ax)
fig.tight_layout();save(fig,'armijo_c')

# backtrack example, exact f=x² x0=2 alpha 1 .5 for c=.25
fig,axs=plt.subplots(1,2,figsize=(10.2,3.55))
a=np.linspace(0,1.25,400);p=(2-4*a)**2;lim=4-.25*16*a
axs[0].plot(a,p,color=BLUE,lw=2.4,label=r'$f(2-4\alpha)$')
axs[0].plot(a,lim,color=RED,ls='--',lw=2,label='граница Армихо')
for aa,txt,col in [(1.,'отклонён',RED),(.5,'принят',GREEN)]:
    axs[0].scatter([aa],[(2-4*aa)**2],s=65,color=col,zorder=5)
    axs[0].annotate(txt,xy=(aa,(2-4*aa)**2),xytext=(aa-.32,4.5 if aa==1 else 2.7),fontsize=10,color=col,arrowprops={'arrowstyle':'->','color':col})
axs[0].set_xlim(0,1.25);axs[0].set_ylim(-1,8.6);axs[0].set_xlabel(r'$\alpha$');axs[0].set_ylabel('значение');axs[0].legend(fontsize=9,loc='upper left');clean(axs[0])
xx=np.linspace(-3.5,3.5,300);axs[1].plot(xx,xx*xx,color=BLUE,lw=2.4)
axs[1].scatter([2,-2,0],[4,4,0],c=[BLUE,RED,GREEN],s=65,zorder=5)
axs[1].annotate(r'$x_0=2$',xy=(2,4),xytext=(2.15,6),arrowprops={'arrowstyle':'->'})
axs[1].annotate(r'$\alpha=1$',xy=(-2,4),xytext=(-3.3,6.8),arrowprops={'arrowstyle':'->','color':RED},color=RED)
axs[1].annotate(r'$\alpha=1/2$',xy=(0,0),xytext=(.35,3.0),arrowprops={'arrowstyle':'->','color':GREEN},color=GREEN)
axs[1].set_xlim(-3.5,3.5);axs[1].set_ylim(-.5,9);axs[1].set_xlabel('$x$');axs[1].set_ylabel('$x^2$');clean(axs[1])
fig.tight_layout();save(fig,'backtrack_example')

# beta and accepted alpha patterns
fig,ax=plt.subplots(figsize=(8.2,3.5));b=np.array([.25,.5,.7,.85]);
for k,val in enumerate(b):
    ns=np.arange(0,9);seq=val**ns
    ax.plot(ns,seq,'-o',lw=1.6,ms=3.7,label=rf'$\beta={val:g}$')
ax.set_ylim(-.02,1.05);ax.set_xlabel('число сокращений');ax.set_ylabel(r'пробный шаг $\alpha_n/\alpha_0$');ax.legend(fontsize=10,ncol=2);clean(ax)
fig.tight_layout();save(fig,'beta_decay')

# compare schedules iterations alpha fixed vs exact vs armijo on quadratic 2d
AA=np.array([[1.,0.],[0.,30.]])
def value(x):return .5*x@(AA@x)
def run(mode,steps=45):
    x=np.array([3.0,1.3]); pts=[x.copy()]; fs=[value(x)]; ev=0
    for k in range(steps):
        g=AA@x
        if mode=='fixed':a=1/30
        elif mode=='exact':a=(g@g)/(g@(AA@g))
        else:
            a=1.0
            while value(x-a*g)>value(x)-.25*a*(g@g): a*=.5;ev+=1
        x=x-a*g;pts.append(x.copy());fs.append(value(x))
    return np.array(pts),np.array(fs),ev
cases={lab:run(code) for lab,code in [('Фиксированный $1/L$','fixed'),('Точный вдоль направления','exact'),('Обратный поиск','armijo')]}
fig,axs=plt.subplots(1,2,figsize=(11.2,3.8))
xx,yy=np.meshgrid(np.linspace(-3.2,3.2,250),np.linspace(-1.4,1.4,240));zz=.5*(xx**2+30*yy**2)
axs[0].contour(xx,yy,zz,levels=[.2,.7,2,4,8,16,30],colors=GRAY,alpha=.33)
for (lab,(pts,fs,_)),color in zip(cases.items(),[BLUE,RED,GREEN]):
    axs[0].plot(pts[:12,0],pts[:12,1],'-o',color=color,ms=2.9,lw=1.5,label=lab)
    axs[1].semilogy(np.maximum(fs,1e-14),color=color,lw=2.2,label=lab)
axs[0].set_xlim(-3.2,3.2);axs[0].set_ylim(-1.4,1.4);axs[0].set_title('Первые 12 шагов');axs[0].set_xlabel('$x_1$');axs[0].set_ylabel('$x_2$');
axs[1].set_title('Убывание целевой функции');axs[1].set_xlabel('итерация');axs[1].set_ylabel('$f(x_k)-f^*$')
axs[1].legend(fontsize=9,loc='upper right');clean(axs[0]);clean(axs[1])
fig.tight_layout();save(fig,'strategy_comparison')

# stopping hazards: tiny step despite high gradient, stationary saddle
fig,axs=plt.subplots(1,2,figsize=(10.2,3.6));xx=np.linspace(-2,2,350)
axs[0].plot(xx,(xx-1)**2,color=BLUE,lw=2.3);p=1.7; q=p-1e-3*2*(p-1)
axs[0].scatter([p,q],[(p-1)**2,(q-1)**2],s=65,c=[RED,GREEN],zorder=4)
axs[0].set_title('Малый шаг: точка почти не меняется',fontsize=11);axs[0].set_xlabel('$x$');axs[0].set_ylabel('$f(x)$');clean(axs[0])
xx=np.linspace(-2,2,200); yy=np.linspace(-2,2,200);xx,yy=np.meshgrid(xx,yy);zz=xx**2-yy**2
axs[1].contour(xx,yy,zz,levels=[-3,-2,-1,-.5,0,.5,1,2,3],colors=BLUE,linewidths=.95)
axs[1].scatter([0],[0],s=75,color=RED,zorder=5);axs[1].annotate(r'$\nabla f(0)=0$',xy=(0,0),xytext=(.35,1.5),arrowprops={'arrowstyle':'->','color':RED})
axs[1].set_title('Градиент нулевой — это может быть седло',fontsize=11);axs[1].set_xlabel('$x_1$');axs[1].set_ylabel('$x_2$');axs[1].set_aspect('equal');clean(axs[1])
fig.tight_layout();save(fig,'stopping_pitfalls')
print('Created',len(list(OUT.glob('*.pdf'))),'vector PDF illustrations.')

# Extra mathematics-oriented figures for the revised lecture.
# 15. Eigenvalue bounds on optimal line-search step (Rayleigh quotient)
fig,ax=plt.subplots(figsize=(8.1,3.25))
mu,L=1.,9.
phi=np.linspace(0,np.pi/2,400)
g=np.column_stack([np.cos(phi),np.sin(phi)])
a=1/(mu*g[:,0]**2+L*g[:,1]**2)
ax.plot(phi,a,c=BLUE,lw=2.6)
ax.axhline(1/L,c=RED,ls='--',lw=1.5,label=r'$1/L$')
ax.axhline(1/mu,c=GREEN,ls='--',lw=1.5,label=r'$1/\mu$')
ax.set_xticks([0,np.pi/4,np.pi/2],labels=[r'$0$',r'$\pi/4$',r'$\pi/2$'])
ax.set_xlim(0,np.pi/2);ax.set_ylim(.05,1.09)
ax.set_xlabel(r'угол $g$ с собственным вектором $v_{\mu}$')
ax.set_ylabel(r'точный шаг $\alpha^*$')
ax.legend(fontsize=10,loc='upper right');clean(ax)
fig.tight_layout();save(fig,'rayleigh_steps')

# 16. Orthogonality: gradient vectors and two subsequent steps
fig,ax=plt.subplots(figsize=(7.2,4.05))
A=np.diag([1.,8.]);xx,yy=np.meshgrid(np.linspace(-2.1,2.2,300),np.linspace(-1.25,1.25,300));Z=.5*(xx**2+8*yy**2)
ax.contour(xx,yy,Z,levels=[.15,.4,.9,1.5,2.5,4,7],colors=BLUE,alpha=.35)
p=np.array([1.75, .92]);g=A@p;a=(g@g)/(g@(A@g));p1=p-a*g;g1=A@p1
a1=(g1@g1)/(g1@(A@g1));p2=p1-a1*g1
ax.plot([p[0],p1[0],p2[0]],[p[1],p1[1],p2[1]],'-o',lw=2.2,color=RED,markersize=5)
ax.annotate('',xy=p1+.15*g1,xytext=p1,arrowprops={'arrowstyle':'->','color':GREEN,'lw':2})
ax.annotate('',xy=p1+.08*g,xytext=p1,arrowprops={'arrowstyle':'->','color':GOLD,'lw':2})
ax.text(p[0]-.1,p[1]+.12,r'$x_k$',fontsize=13)
ax.text(p1[0]+.08,p1[1]-.18,r'$x_{k+1}$',fontsize=13)
ax.text(p2[0]-.4,p2[1]+.12,r'$x_{k+2}$',fontsize=13)
ax.text(-1.8,1.07,r'$g_k^\top g_{k+1}=0$',color=RED,fontsize=14)
ax.set_xlim(-2.1,2.2);ax.set_ylim(-1.25,1.25);ax.set_aspect('equal');ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');clean(ax)
fig.tight_layout();save(fig,'orthogonal_gradients')

# 17. Angle with negative gradient controls admissible Armijo step
fig,ax=plt.subplots(figsize=(7.8,3.1))
theta=np.linspace(0,np.pi/2,400)
ax.plot(theta,np.cos(theta),lw=2.5,c=BLUE)
ax.axhline(0,c=GRAY,lw=.8)
ax.set_xticks([0,np.pi/6,np.pi/3,np.pi/2],labels=[r'$0$',r'$\pi/6$',r'$\pi/3$',r'$\pi/2$'])
ax.set_xlim(0,np.pi/2);ax.set_ylim(-.02,1.08)
ax.set_xlabel(r'угол $\theta$ между $d$ и $-\nabla f(x)$')
ax.set_ylabel(r'$\cos\theta$')
ax.set_title('Почти перпендикулярное градиенту направление требует осторожного шага',fontsize=10.6)
clean(ax);fig.tight_layout();save(fig,'angle_armijo')

# 18. Gradients stationarity convergence from telescoping bound
fig,ax=plt.subplots(figsize=(8.2,3.35))
k=np.arange(1,301)
for normval,color,lab in [(1.,BLUE,r'$C/\sqrt{N}$'),(.4,GREEN,r'$0.4C/\sqrt{N}$')]:
    ax.plot(k,normval/np.sqrt(k),lw=2,color=color,label=lab)
ax.set_xlabel('число итераций $N$');ax.set_ylabel('верхняя оценка на min норму градиента')
ax.set_title('Оценка на лучшую норму градиента среди первых $N$ шагов',fontsize=11)
ax.legend(fontsize=10);clean(ax);fig.tight_layout();save(fig,'stationarity_rate')
