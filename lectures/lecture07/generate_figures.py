"""Иллюстрации лекции 7. Запуск: python generate_figures.py
Только NumPy и Matplotlib; пригодно для Google Colab.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
BLUE='#003D7C'; RED='#D73256'; TEAL='#168E91'; GRAY='#66768A'; LIGHT='#EAF2FB'; PALE='#FFF2F4'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                     'font.size':11,'axes.titlesize':11,'figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.05,dpi=150)
    plt.close(fig)

def box(ax,x,y,label,w=1.22,h=.56,accent=BLUE,active=False,fs=13):
    fill=PALE if active else LIGHT
    patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.08,rounding_size=0.09',
                         facecolor=fill,edgecolor=RED if active else accent,linewidth=1.8 if active else 1.1)
    ax.add_patch(patch)
    ax.text(x,y,label,ha='center',va='center',fontsize=fs,color=BLUE,zorder=4)
    if not hasattr(ax,'_boxes'): ax._boxes={}
    ax._boxes[(x,y)]=(w,h)

def connect(ax,p,q,color=GRAY,label=None,rad=0,fs=9):
    # Начало/конец стрелки лежат за пределами прямоугольников,
    # а не поверх текста (важно для схем в стиле ВШЭ).
    import numpy as _np
    d=_np.array(q,dtype=float)-_np.array(p,dtype=float)
    if _np.linalg.norm(d)<1e-8: return
    def boundary(center,direction):
        w,h=ax._boxes.get(tuple(center),(0,0))
        rx=w/2+.105;ry=h/2+.105
        scales=[]
        if abs(direction[0])>1e-8:scales.append(rx/abs(direction[0]))
        if abs(direction[1])>1e-8:scales.append(ry/abs(direction[1]))
        return _np.array(center)+direction*min(scales)
    start=boundary(p,d);end=boundary(q,-d)
    a=FancyArrowPatch(tuple(start),tuple(end),arrowstyle='-|>',mutation_scale=12,
                      linewidth=1.45,color=color,shrinkA=2,shrinkB=3,
                      connectionstyle=f'arc3,rad={rad}',zorder=2)
    ax.add_patch(a)
    if label:
        m=(start+end)/2
        ax.text(m[0],m[1]+.17,label,ha='center',va='bottom',fontsize=fs,
                color=color,bbox=dict(facecolor='white',edgecolor='none',pad=1.2),zorder=5)

def graph(fig_name,active=(),reverse=False,values=False,annot=False):
    fig,ax=plt.subplots(figsize=(10,4.8))
    # Единая топология на всех рисунках: y слева, x справа.
    pos={'y':(1.2,3.85),'x':(6.35,3.85),
         'a':(2.65,2.65),'b':(6.35,2.65),
         'c':(4.45,1.40),'f':(4.45,.15)}
    titles={'x':'$x$','y':'$y$','a':'$a=xy$','b':'$b=\\sin x$','c':'$c=a+b$','f':'$f=c^2$'}
    if values:
        titles.update({'x':'$x=1$','y':'$y=2$','a':'$a=2$','b':'$b=\\sin 1$','c':r'$c=2+\sin 1$','f':r'$f=(2+\sin 1)^2$'})
    if annot:
        titles.update({'f':r'$\bar f=1$','c':r'$\bar c=2c$'})
        if any(n in active for n in ('a','b','x','y')):
            titles.update({'a':r'$\bar a=2c$','b':r'$\bar b=2c$'})
        if 'x' in active or 'y' in active:
            titles.update({'x':r'$\bar x$','y':r'$\bar y$'})
    for k,(x,y) in pos.items():
        w=1.17 if k in ('x','y') else (2.5 if values and k=='f' else 2.0 if k in ('b','c','f') else 1.5)
        box(ax,x,y,titles[k],w=w,h=.60,active=k in active,fs=12.5)
    ed=[('y','a'),('x','a'),('x','b'),('a','c'),('b','c'),('c','f')]
    for u,v in ed:
        p,q=(pos[v],pos[u]) if reverse else (pos[u],pos[v])
        col=RED if (u in active or v in active) else GRAY
        connect(ax,p,q,color=col,rad=0.02 if (u,v)==('x','a') else 0)
    ax.set_xlim(-.1,8.2);ax.set_ylim(-.48,4.5);ax.axis('off')
    save(fig,fig_name)

# 1. Composition grows but keeps a compact graph
fig,ax=plt.subplots(figsize=(9.5,2.45))
xs=[.9,2.65,4.55,6.50,8.35]; lbl=['$x$','$x^2+e^x$','$\\sin(\\cdot)$','$(\\cdot+x^3)^2$','$f(x)$']
for i,(x,s) in enumerate(zip(xs,lbl)):
    box(ax,x,1.18,s,w=(1.10 if i==4 else 1.68 if i>0 else .85),h=.7,active=i==3,fs=11)
    if i:connect(ax,(xs[i-1],1.18),(x,1.18))
ax.set_xlim(.1,9.3);ax.set_ylim(.3,2.0);ax.axis('off');save(fig,'long_composition')

# 2. Differential is linear tangent approximation
fig,ax=plt.subplots(figsize=(7.6,3.7))
x=np.linspace(-1.8,2.2,450);f=.7*x*x+.25*x+1;x0=.45;h=.72
ax.plot(x,f,color=BLUE,lw=2.4);tan=(.7*x0*x0+.25*x0+1)+(1.4*x0+.25)*(x-x0)
ax.plot(x,tan,color=RED,lw=1.9,ls='--');ax.scatter([x0,x0+h],[np.interp(x0,x,f),np.interp(x0+h,x,f)],c=[BLUE,BLUE],s=43,zorder=3)
ax.annotate('',(x0+h,.44),(x0,.44),arrowprops={'arrowstyle':'<->','color':TEAL,'lw':1.5})
ax.text(x0+h/2,.25,'$dx$',ha='center',color=TEAL)
ax.annotate('',(x0+h,np.interp(x0+h,x,tan)),(x0+h,np.interp(x0,x,f)),arrowprops={'arrowstyle':'<->','color':RED,'lw':1.5})
ax.set_xlim(-1.7,2.1);ax.set_ylim(0,4.5);ax.set_xlabel('$x$');ax.set_ylabel('$f(x)$');ax.grid(alpha=.13)
ax.legend(['$f(x)$','линейная часть'],loc='upper left',frameon=False);save(fig,'differential_tangent')

# 3. Dimensions in matrix multiplication (diagram)
fig,ax=plt.subplots(figsize=(9.2,2.5))
for x,w,t in [(1.3,2.2,'$A_{m\\times d}$'),(4.35,1.7,'$dx_{d\\times1}$'),(7.2,2.0,'$dz_{m\\times1}$')]:box(ax,x,1.2,t,w=w,h=.85,fs=20)
ax.text(2.9,1.2,'$\\cdot$',ha='center',va='center',fontsize=23,color=BLUE)
ax.text(5.65,1.2,'$=$',ha='center',va='center',fontsize=23,color=BLUE)
ax.set_xlim(-.1,8.6);ax.set_ylim(.1,2.25);ax.axis('off');save(fig,'matrix_dimensions')

# 4. Least squares residual -> gradient
fig,ax=plt.subplots(figsize=(10,2.7))
xx=[1.2,3.8,6.2,8.75];ll=['$w$','$r=Xw-y$','$X^\\top r$','$\\nabla F(w)$']
for i,x in enumerate(xx):box(ax,x,1.3,ll[i],w=1.9 if i else 1.05,h=.78,active=i==3,fs=14)
for i in range(3):connect(ax,(xx[i],1.3),(xx[i+1],1.3))
ax.text(3.8,.35,'ошибки объектов',ha='center',color=GRAY,fontsize=10)
ax.text(6.2,.35,'вклад в параметры',ha='center',color=GRAY,fontsize=10)
ax.set_xlim(.05,10);ax.set_ylim(.03,2.3);ax.axis('off');save(fig,'least_squares_pipeline')

# 5. Chain rule: dimensions
fig,ax=plt.subplots(figsize=(9.5,2.3))
for x,w,t in [(1.0,1.35,'$x\\in\\mathbb{R}^d$'),(4.15,1.65,'$g(x)\\in\\mathbb{R}^m$'),(7.55,1.7,'$h(g(x))\\in\\mathbb{R}^p$')]:box(ax,x,1.15,t,w=w,h=.67,fs=14)
connect(ax,(1.0,1.15),(4.15,1.15),label='$J_g: m\\times d$')
connect(ax,(4.15,1.15),(7.55,1.15),label='$J_h: p\\times m$')
ax.set_xlim(-.15,8.85);ax.set_ylim(.1,2.15);ax.axis('off');save(fig,'chain_dimensions')

# 6. Graph states (highlighted and reverse)
graph('graph_overview')
graph('graph_forward_start',active=('x','y'))
graph('graph_forward_middle',active=('a','b'))
graph('graph_forward_values',active=('c','f'),values=True)
graph('graph_local_derivatives',active=('a','b','c'))
graph('graph_reverse_seed',active=('f','c'),reverse=True,annot=True)
graph('graph_reverse_mid',active=('a','b'),reverse=True,annot=True)
graph('graph_reverse_inputs',active=('x','y'),reverse=True,annot=True)

# 7. Two paths from x to f — keep y on the left, x on the right
fig,ax=plt.subplots(figsize=(10,3.9))
p={'y':(1.0,3.1),'x':(6.6,3.1),'a':(2.7,2.0),'b':(6.6,2.0),'c':(4.5,.9),'f':(4.5,-.22)}
for k,v in p.items():
    title={'y':'$y$','x':'$x$','a':'$a=xy$','b':'$b=\\sin x$','c':'$c=a+b$','f':'$f=c^2$'}[k]
    box(ax,*v,title,w=1.65 if k in ('a','b','c','f') else 1.0,active=k in ('x','a','b'))
for u,v in [('y','a'),('x','a'),('x','b'),('a','c'),('b','c'),('c','f')]:
    connect(ax,p[u],p[v],color=RED if u in ('x','a','b','c') else GRAY)
ax.set_xlim(-.15,9.25);ax.set_ylim(-.64,3.75);ax.axis('off');save(fig,'two_paths')

# 8. Forward/reverse concept diagram (both rows)
fig,axs=plt.subplots(2,1,figsize=(9.4,3.7))
for idx,ax in enumerate(axs):
    X=[.9,3.45,6.1,8.65];ss=['входы','операции','промежуточные','потеря']
    for j,(x,s) in enumerate(zip(X,ss)):box(ax,x,.55,s,w=1.85,h=.53,active=(idx==0 and j==0)or(idx==1 and j==3),fs=11)
    for j in range(3):
        p,q=((X[j],.55),(X[j+1],.55)) if idx==0 else ((X[j+1],.55),(X[j],.55))
        connect(ax,p,q,color=TEAL if idx==0 else RED)
    ax.set_xlim(-.18,9.8);ax.set_ylim(-.15,1.2);ax.axis('off')
axs[0].set_title('Прямой режим: значения и направленные производные',fontsize=12,color=BLUE)
axs[1].set_title('Обратный режим: производные от выхода к входам',fontsize=12,color=BLUE)
fig.subplots_adjust(hspace=.6);save(fig,'forward_reverse_compare')

# 9. FD error curve and analytic numerical evaluation
h=np.logspace(-16,-.4,230);x0=.7;truth=np.cos(x0)
e=np.abs((np.sin(x0+h)-np.sin(x0-h))/(2*h)-truth)
fig,ax=plt.subplots(figsize=(7.8,4.1));ax.loglog(h,e+1e-17,lw=2.25,color=BLUE)
ax.axvline(h[np.argmin(e)],lw=1.3,ls='--',color=RED)
ax.text(2e-13,2e-3,'округление',color=GRAY);ax.text(3e-3,3e-6,'погрешность\nприближения',color=GRAY)
ax.set_xlabel('шаг $h$');ax.set_ylabel(r'$|D_h\sin(x)-\cos(x)|$');ax.grid(alpha=.18,which='both');save(fig,'finite_difference_error')

print('Сгенерировано иллюстраций:',len(list(OUT.glob('*.pdf'))))
