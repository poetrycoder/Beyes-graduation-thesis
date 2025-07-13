#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#实验" data-toc-modified-id="实验-1">实验</a></span><ul class="toc-item"><li><span><a href="#1.1" data-toc-modified-id="1.1-1.1">1.1</a></span></li><li><span><a href="#1.2" data-toc-modified-id="1.2-1.2">1.2</a></span></li><li><span><a href="#1.3" data-toc-modified-id="1.3-1.3">1.3</a></span></li><li><span><a href="#2.1" data-toc-modified-id="2.1-1.4">2.1</a></span></li><li><span><a href="#2.2" data-toc-modified-id="2.2-1.5">2.2</a></span></li><li><span><a href="#2.3" data-toc-modified-id="2.3-1.6">2.3</a></span></li><li><span><a href="#3" data-toc-modified-id="3-1.7">3</a></span></li><li><span><a href="#4" data-toc-modified-id="4-1.8">4</a></span></li></ul></li></ul></div>

# ## 实验

# ### 1.1

# In[2]:


from empiricaldist import Pmf
from scipy.stats import binom


# In[3]:


prior = Pmf([0.7, 0.3], ['theta_1', 'theta_2'])
prior


# In[4]:


likelihood = binom.pmf(2, 8, [0.1, 0.2])
likelihood


# In[5]:


posterior = prior * likelihood
posterior.normalize()


# In[6]:


posterior


# ### 1.2

# In[1]:


from empiricaldist import Pmf
from scipy.stats import poisson


# In[2]:


prior = Pmf([0.4, 0.6], ['lambda_1', 'lambda_2'])
prior


# In[3]:


likelihood = poisson.pmf(3, [1.0, 1.5])
likelihood


# In[4]:


posterior = prior * likelihood
posterior.normalize()


# In[5]:


posterior


# ### 1.3

# In[1]:


import numpy as np
from empiricaldist import Pmf
from scipy.stats import binom


# In[15]:


qs = np.linspace(0, 1, 100)
prior = Pmf.from_seq(qs)  ## 根据数组等可能生成 Pmf 
# prior


# In[3]:


## 等可能生成非规范 Pmf 然后再规范化
# prior = Pmf(1, qs)
# prior.normalize()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid', font = 'SimHei', font_scale = 1)
plt.figure(figsize = (6, 4))

plt.plot(prior)

plt.xlabel(''r'$\theta$')
plt.ylabel('PDF')
plt.title('先验分布')
plt.show()


# In[5]:


likelihood = binom.pmf(3, 8, qs)


# In[8]:


posterior = prior*likelihood
posterior.normalize()
# posterior


# In[9]:


plt.plot(posterior)

plt.xlabel(''r'$\theta$')
plt.ylabel('PDF')
plt.title('后验分布')
plt.show()


# In[10]:


qs = np.linspace(0, 1, 100)
ps = 2*(1 - qs)
prior = Pmf(ps, qs)
prior.normalize()


# In[11]:


plt.plot(prior)

plt.xlabel(''r'$\theta$')
plt.ylabel('PDF')
plt.title('先验分布')
plt.show()


# In[13]:


likelihood = binom.pmf(3, 8, qs)

posterior = prior*likelihood
posterior.normalize()
# posterior


# In[14]:


plt.plot(posterior)

plt.xlabel(''r'$\theta$')
plt.ylabel('PDF')
plt.title('后验分布')
plt.show()


# ### 2.1

# In[1]:


from scipy.stats import gamma

mu, sd = 0.2, 1.0
alpha, beta = (mu/sd)**2, mu/sd**2
alpha, beta


# In[2]:


def make_gamma(alpha, beta):
    """ 构造 gamma 分布对象 均值为 alpha/beta 型 """
    dist = gamma(alpha, scale = 1/beta)
    dist.alpha = alpha
    dist.beta = beta
    return dist


# In[3]:


prior_gamma = make_gamma(alpha, beta)
prior_gamma.mean()  ## alpha/beta


# In[4]:


def update_gamma(prior, data):
    """ 更新 gamma 先验 """
    k, t = data
    alpha = prior.alpha + k
    beta = prior.beta + t
    return make_gamma(alpha, beta)


# In[9]:


data = [20, 20*3.8]
posterior_gamma = update_gamma(prior_gamma, data)
posterior_gamma.mean()


# In[10]:


posterior_gamma.ppf([0.03, 0.97])


# In[11]:


alpha_pos = posterior_gamma.alpha
beta_pos = posterior_gamma.beta
alpha_pos, beta_pos


# In[12]:


from scipy.stats import invgamma

invgamma(alpha_pos, scale = beta_pos).mean()


# ### 2.2

# In[1]:


import numpy as np
from empiricaldist import Pmf
from scipy.stats import gamma, poisson

gamma_dist = gamma(3, scale = 1/1)
lam = np.linspace(0, 8, 100)


# In[2]:


def pmf_from_dist(dist, qs):
    """ 连续分布的离散近似 """
    ps = dist.pdf(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf

prior = pmf_from_dist(gamma_dist, lam)


# In[14]:


likelihood = poisson(lam).pmf(2) * poisson(lam).pmf(2) * poisson(lam).pmf(6)* poisson(lam).pmf(0) * poisson(lam).pmf(3)

posterior = prior * likelihood
posterior.normalize()
# posterior


# In[15]:


posterior.mean()


# In[6]:


posterior.var()


# In[16]:


posterior.credible_interval(0.94)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid', font = 'SimHei', font_scale = 1)
plt.figure(figsize = (6, 4))

plt.plot(prior, label = '先验', color = 'b')
plt.plot(posterior, label = '后验', color = 'g')
plt.vlines(prior.mean(), 0, 0.05, label = '先验均值', linestyles = '-.', color = 'b')
plt.vlines(posterior.mean(), 0, 0.05, label = '后验均值', linestyles = ':', color = 'g')

plt.xlabel(''r'$\lambda$')
plt.ylabel('PDF')
plt.title('先验与后验分布')
plt.legend()
plt.show()


# ### 2.3

# In[2]:


import pymc3 as pm
from scipy.stats import mode

with pm.Model() as model_1:    
    theta = pm.Beta('theta', alpha = 5, beta = 10)
    obs = pm.Binomial('obs', n = 20, p = theta, observed = [3])
    trace_1 = pm.sample(500)
    burned_trace_1 = trace_1[200:]


# In[3]:


theta_sample_1 = burned_trace_1['theta']
theta_sample_1.mean()


# In[4]:


mode(theta_sample_1)[0][0]


# In[5]:


with pm.Model() as model_2:    
    theta = pm.Beta('theta', alpha = 5, beta = 10)
    obs = pm.Binomial('obs', n = 20, p = theta, observed = [3, 0])
    trace_2 = pm.sample(500)
    burned_trace_2 = trace_2[200:]


# In[6]:


theta_sample_2 = burned_trace_2['theta']
theta_sample_2.mean()


# In[7]:


mode(theta_sample_2)[0][0]


# ### 3

# 收益函数

# $$Q(\theta ,a) = \left\{ {\begin{array}{*{20}{l}}
# {(45 - 8) \times 0.8a - 8 \times 0.2a = 28a}&{0.8a \le \theta }\\
# {(45 - 8) \times \theta  - 8 \times 0.2a - (0.8a - \theta ) \times (8 - 5) = 40\theta  - 4a}&{0.8a > \theta }
# \end{array}} \right.$$

# 先验期望收益

# $\begin{array}{*{20}{l}}
# {\bar Q(a) = \sum\limits_{\theta  \in \Theta }^{} {Q(\theta ,a){\rm{\pi }}(\theta )} }\\
# { = \sum\limits_{k = 0}^{[0.8a] - 1} {(40k - 4a) \cdot P(\theta  = k)}  + \sum\limits_{k = [0.8a]}^{ + \infty } {28a \cdot P(\theta  = k)} }\\
# { = \sum\limits_{k = 0}^{[0.8a] - 1} {(40k - 4a) \cdot P(\theta  = k)}  - \sum\limits_{k = 0}^{[0.8a] - 1} {28a \cdot P(\theta  = k)}  + \sum\limits_{k = 0}^{ + \infty } {28a \cdot P(\theta  = k)} }\\
# { = \sum\limits_{k = 0}^{[0.8a] - 1} {(40k - 32a)P(\theta  = k)}  + 28a\sum\limits_{k = 0}^{ + \infty } {P(\theta  = k)} }\\
#  = {\sum\limits_{k = 0}^{[0.8a] - 1} {(40k - 32a)\frac{{{\lambda ^k}}}{{k!}}{e^{ - \lambda }}}  + 28a}
# \end{array}$

# In[16]:


## 先验期望收益 -- 泊松先验分布
import numpy as np
from scipy.stats import poisson
def Q_bar(a):
    t = round(0.8*a) - 1
    k = np.arange(t + 1)
    return  sum((40*k - 32*a)*poisson.pmf(k, 15)) + 28*a
m = ()
for i in range(41) :
    m = np.append(m, Q_bar(i))
print("先验期望收益：\n", np.round(m, 2))
print("最大平均收益%.2f 最优决策a=%d" %(max(m), np.where(m == max(m))[0]))


# 损失函数：当${0.8a = \theta }$时得到最大收益${35\theta }$

# $$ L(\theta ,a) = \left\{ {\begin{array}{*{20}{l}}
# {35\theta  - 28a}&{0.8a \le \theta }\\
# {35\theta  - (40\theta  - 4a) = 4a - 5\theta }&{0.8a > \theta }
# \end{array}} \right. $$

# 先验期望损失

# $\begin{array}{*{20}{l}}
# {\bar L(a) = \int_\Theta ^{} {L(\theta ,a){\rm{\pi }}(\theta ){\rm{d}}\theta } }\\
# { = \int_0^{0.8a} {(4a - 5\theta ) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta }  + \int_{0.8a}^{ + \infty } {(35\theta  - 28a) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta } }\\
# { = \int_0^{0.8a} {(4a - 5\theta ) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta }  - \int_0^{0.8a} {(35\theta  - 28a) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta }  + \int_0^{ + \infty } {(35\theta  - 28a) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta } }\\
# \begin{array}{l}
#  = \int_0^{0.8a} {(32a - 40\theta ) \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta }  + 35\int_0^{ + \infty } {\theta  \cdot {\rm{\pi }}(\theta ){\rm{d}}\theta }  - 28a\int_0^{ + \infty } {{\rm{\pi }}(\theta ){\rm{d}}\theta } \\
#  = 32a \times Ga(0.8a,30,2) - 40\frac{{{2^{30}}}}{{\Gamma (30)}}\frac{{\Gamma (31)}}{{{2^{31}}}}\int_0^{0.8a} {\frac{{{2^{31}}}}{{\Gamma (31)}}{\theta ^{31 - 1}}{e^{ - 2\theta }}{\rm{d}}\theta }  + 35 \times 15 - 28a
# \end{array}\\
# { = 32a \times Ga(0.8a,30,2) - 40 \times 15 \times Ga(0.8a,31,2) + 35 \times 15 - 28a}
# \end{array}$
# 
# 其中 Ga(x,a,b) 表示参数(a,b)的伽玛分布函数在x处的取值

# In[18]:


## 先验期望损失 -- 伽玛先验分布 -- 离散观测点
import numpy as np
from scipy.stats import gamma
def L_bar(a):    
    return 32*a*gamma.cdf(0.8*a, 30, scale = 1/2) - 600*gamma.cdf(0.8*a, 31, scale = 1/2) + 35*15 - 28*a
m = ()
for i in range(51) :
    m = np.append(m, L_bar(i))
print("先验期望损失：\n", np.round(m, 2))
print("最小平均损失%.2f 最优决策a=%d" %(min(m), np.where(m == min(m))[0]))


# ### 4

# 收益函数与损失函数

# $\begin{array}{l}
# Q(\theta ,a) = \left\{ {\begin{array}{*{20}{l}}
# {\theta  \times 10 \times 10000 - 400000}&{a = {a_1}}\\
# 0&{a = {a_2}}
# \end{array}} \right.\\
# \begin{array}{*{20}{c}}
# {L(\theta ,{a_1}) = \left\{ {\begin{array}{*{20}{l}}
# {100000 \times (4 - \theta )}&{\theta  \le {\theta _0}}\\
# 0&{\theta  > {\theta _0}}
# \end{array}} \right.}&{L(\theta ,{a_2}) = \left\{ {\begin{array}{*{20}{l}}
# 0&{\theta  \le {\theta _0}}\\
# {100000 \times (\theta  - 4)}&{\theta  > {\theta _0}}
# \end{array}} \right.}
# \end{array}
# \end{array}$
# 
# 其中平衡点${\theta _0} = 4$

# 先验期望收益

# ${\bar Q({a_1}) = {E^\theta }(\theta ) \times 100000 - 400000 = 3.9 \times 100000 - 400000 =  - 100000}$
# 
# ${\bar Q({a_2}) = 0}$
# 
# ${\bar Q({a_1}) < \bar Q({a_2})}$，不应该进行收购，最优行动${a' = {a_2}}$

# In[27]:


import numpy as np
from empiricaldist import Pmf
from scipy.stats import norm

norm_dist = norm(3.9, 0.8)
theta = np.arange(-7.9, 8.0, 0.1)

def pmf_from_dist(dist, qs):
    ps = dist.pdf(qs)
    pmf = Pmf(ps, qs)
    pmf.normalize()
    return pmf

prior = pmf_from_dist(norm_dist, theta)
# prior


# 最优行动损失函数

# $L(\theta ,a') = \left\{ {\begin{array}{*{20}{l}}
# 0&{\theta  \le 4}\\
# {100000 \times (\theta  - 4)}&{\theta  > 4}
# \end{array}} \right.$

# In[28]:


def L(theta):    
    if theta <= 4:
        return 0
    else:
        return 100000*(theta - 4)

L_a = [L(i) for i in theta]
# L_a


# In[29]:


pri_EVPI = np.dot(L_a, prior)
pri_EVPI


# $抽样分布：\begin{array}{*{20}{c}}
# {{{\bar X}_{100}}{\rm{\~}}N(\theta ,\frac{4}{{100}})}&{\bar x = 4.1}
# \end{array}$

# In[31]:


likelihood = norm(theta, np.sqrt(4/100)).pdf(4.1)
# likelihood


# In[32]:


posterior = prior * likelihood
posterior.normalize()
# posterior


# In[33]:


pos_EVPI = np.dot(L_a, posterior)
pos_EVPI


# In[35]:


mar = sum(prior * likelihood)
mar


# In[36]:


pos_EVPI_Exp = pos_EVPI * mar
pos_EVPI_Exp


# In[37]:


EVSI = pri_EVPI - pos_EVPI_Exp
EVSI


# In[38]:


ENGS_100 = EVSI - 25*100
print("抽样净益：%.2f" %ENGS_100)
if ENGS_100 > 0:
    print("值得抽样")
else:
    print("不值得抽样")


# In[39]:


def Q(theta, a):
    if a == 'a1':
        return 100000*(theta - 4)
    else :        
        return [0 for i in range(len(theta))]

# Q(theta, 'a1')
# Q(theta, 'a2')


# In[40]:


def pos_Q_bar(a):
    return np.dot(Q(theta, a), posterior)
    
# pos_Q_bar('a1')
# pos_Q_bar('a2')


# In[41]:


if pos_Q_bar('a1') > pos_Q_bar('a2'):
    print("最优行动为a1,应该进购")
else :
    print("最优行动为a2,不应该进购")

