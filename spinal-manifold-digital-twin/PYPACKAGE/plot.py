import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
import random
from PYPACKAGE.parula_color import parula
from sklearn.manifold import Isomap


def plot_data(type, time=None, bp=None, fr=None, Hz=None, iso_0=None, iso_1=None, iso_2=None, color=None, L=None) :
    
    if type == "full_bp_plot" :
        print(type)
        t = time
        bp_data = bp
        
        if color == "BP" : 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(t, bp_data, c = bp_data, cmap = parula())
        elif color == "TIME" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.scatter(t, bp_data, c = t, cmap = parula())
        elif color == "ORGINAL" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.plot(t, bp_data)
    
    elif type == "Ref_bp_plot" :
        print(type)
        t = time
        bp_data = bp
        
        if color == "BP" : 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(t, bp_data, c = bp_data, cmap = 'gist_heat')
        elif color == "TIME" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.scatter(t, bp_data, c = t, cmap = parula())
        elif color == "ORGINAL" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.plot(t, bp_data, color='red', linewidth=5)
        elif color == "ISO" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.scatter(t, bp_data, c = iso_0, cmap = parula())
                            
    elif type == "Ref_fr_plot" :
        # print(type)
        tRef = time
        frRef = fr
        bpRef = bp
        
        if color == "BP" : 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(tRef, frRef, c = bpRef, cmap = parula())
        elif color == "TIME" : 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(tRef, frRef, c = tRef, cmap = parula())        
        elif color == "ORGINAL" :
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.plot(tRef, frRef)
   
    elif type == "Isomap 2d & 3d" :
        # print(type)
        fr_isomap_0 = iso_0
        fr_isomap_1 = iso_1
        fr_isomap_2 = iso_2
        select_freq = Hz
        bpRef = bp
        tRef = time
        
        if color == "BP" :
            fig = plt.figure()
            ax_iso_3d = fig.add_subplot(111, projection = '3d')
            ax_iso_3d.scatter(fr_isomap_0,fr_isomap_1,fr_isomap_2, c=bpRef, cmap = 'gist_heat')#, c=t%L, cmap = parula())
            fig = plt.figure()
            ax_iso_3d.axis('off')  
            ax_iso = fig.add_subplot(111)
            ax_iso.scatter(fr_isomap_0,fr_isomap_1, label=select_freq , c=bpRef, cmap = parula())#, c=t%L, cmap = parula())

        elif color == "TIME" :
            fig = plt.figure()
            ax_iso_3d = fig.add_subplot(111, projection = '3d')
            ax_iso_3d.scatter(fr_isomap_0,fr_isomap_1,fr_isomap_2, c=tRef, cmap = parula())#, c=t%L, cmap = parula())
            fig = plt.figure()  
            ax_iso_3d.axis('off')  
            
            ax_iso = fig.add_subplot(111)
            ax_iso.scatter(fr_isomap_0,fr_isomap_1, label=select_freq , c=tRef, cmap = parula())#, c=t%L, cmap = parula())
            # ax_iso.scatter(fr_isomap_0,fr_isomap_1, label=select_freq , c=tRef, cmap = parula())#, c=t%L, cmap = parula())

            # 1/4 지점과 3/4 지점에 빨간색으로 표시된 추가적인 점 찍기
            quarter1_idx = int(len(tRef) / 4)
            quarter3_idx = int(3 * len(tRef) / 4)

            ax_iso.scatter(fr_isomap_0[quarter1_idx], fr_isomap_1[quarter1_idx], c='black', label=f'1/4 Point: {tRef[quarter1_idx]:.2f}')
            ax_iso.scatter(fr_isomap_0[quarter3_idx], fr_isomap_1[quarter3_idx], c='red', label=f'3/4 Point: {tRef[quarter3_idx]:.2f}')

            print(tRef)
            
        elif color == "FANCY" :
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # # scatter 입체감 강조: 큰 점, 약간의 투명도, black edge
            # sc = ax.scatter(fr_isomap_0, fr_isomap_1, fr_isomap_2,
            #                 c=tRef, cmap=parula(), s=30, edgecolors='k', alpha=0.9, linewidths=0.3)
            sc = ax.scatter(fr_isomap_0, fr_isomap_1, fr_isomap_2,
                c=bpRef, cmap='gist_heat', s=90, edgecolors='black', alpha=0.8, linewidths=0.2)
                # c=tRef, cmap=parula(), s=80, edgecolors='black', alpha=0.9, linewidths=0.2)

            # 카메라 뷰 조정으로 입체감 강조
            ax.view_init(elev=30, azim=45)  # 위에서 약간 비스듬히 바라보는 시점

            # 축 지우기 (더 깔끔하게)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()

            # 컬러바 (선택)
            fig.colorbar(sc, ax=ax, shrink=0.5, label='Time (sec)')

            plt.tight_layout()
            plt.show()
        
        elif color == "ORGINAL" :
            fig = plt.figure()  
            ax_iso_3d = fig.add_subplot(111, projection = '3d')
            ax_iso_3d.scatter(fr_isomap_0,fr_isomap_1,fr_isomap_2)
            fig = plt.figure()  
            ax_iso = fig.add_subplot(111)
            ax_iso.scatter(fr_isomap_0,fr_isomap_1, label=select_freq)