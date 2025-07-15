#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
x = np.array([100, 50, 10, 30, 40, 9, -60, -20, 6])
BeMa = pd.DataFrame(x.reshape(3, 3), 
                               columns = ['行动'+str(i+1)for i in range(3)],
                                index = ['状态'+str(j+1)for j in range(3)])
print("收益矩阵：\n", BeMa)


# In[2]:


print("|---决策准则清单---|")
print("|---1：悲观准则---|")
print("|---2：乐观准则---|")
print("|---3：折中准则---|")
print("|---4：先验期望准则---|")
print("|---0：退出选择---|")


# In[3]:


choice = input("请选择决策准则(0-4):")
if choice=='1':
    print("悲观准则：最优行动-->\'%s\'"
         %(BeMa.columns[np.argmax(BeMa.min())]))
elif choice=='2':
    print("乐观准则：最优行动-->\'%s\'"
         %(BeMa.columns[np.argmax(BeMa.max())]))
elif choice=='3':
    alpha = float(input("请输入乐观系数："))
    Ha = alpha*BeMa.max()+(1-alpha)*BeMa.min()
    print("折中准则：乐观系数取%.2f时的最优行动-->\'%s\'"
         % (alpha, BeMa.columns[np.argmax(Ha)]))
elif choice=='4':
        pri=input('请输入先验分布以英文逗号分隔：')
        prisp = pri.split(",")
        PRI = [float(prisp[i]) for i in range(len(prisp))]
        peb = BeMa.apply(lambda x: np.dot(x, PRI))
        print("先验期望准则：先验分布取%s时的最优行动-->\'%s\'"
             %(prisp, BeMa.columns[np.argmax(peb)]))
elif choice=='0':
    print("退出选择")
else:
    print("请输入0-4的数字！")


# In[ ]:




