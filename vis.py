import os
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self):
        track_colors = [
            [220, 0, 0], [40, 190, 35], [150, 100, 220],
            [250, 150, 100], [30, 30, 20], [0, 100, 220],

            [170, 60, 80], [150, 190, 35], [150, 230, 220],
            [220, 200, 70], [120, 120, 120], [40, 220, 150], # 여기까지는 비슷한거 없음..?

            [175, 80, 100], [50, 120, 190], [120, 220, 100],
            [180, 30, 180], [50, 200, 200], [185, 125, 150],

            [75, 80, 100], [150, 120, 190], [120, 220, 200],
            [80, 30, 180], [150, 200, 200], [185, 125, 250],

            [135, 180, 30], [102, 220, 10], [120, 20, 10],
            [15, 55, 220], [50, 20, 20], [15, 125, 50],
        ]

        self.RGB = np.zeros((len(track_colors), 3))

        for ci, color in enumerate(track_colors):
            self.RGB[ci] = [c/255. for c in color]

    def draw_presence_bar(self, ax, title_name, gt, pred_list):
        height = 0.5
        yticks = ['GT', 'Pred_Bbox']
        
        legend_name = ['No Presence', 'Presence']
        last_idx = len(gt)
        
        data = self._merge_data(gt, pred_list)
        print(data.shape)
        pos_info = self._find_pos_list(data)
        print(len(pos_info))
        # print(pos_info[0])
        
        for i, tick_name in enumerate(yticks):
            for pos in pos_info[i]:
                print(i, len(pos_info[i]), pos)
                ax.barh(y=i+1, width=pos[1]-pos[0], left=pos[0], height=height, color=self.RGB[int(pos[-1])])
                
        ax.set_title(title_name, fontsize=30)
        
        #### 6. x축 세부설정
        step_size = 720 #WINDOW_SIZE # xtick step_size
        ax.set_xticks(range(0, len(gt), step_size)) # step_size
        
        frame_label = [str(i) for i in range(len(gt))]
        time_label = [str(i // 3600) + ':' + str(i % 3600 // 60) + ':' + str(i % 3600 % 60) for i in range(len(gt))]
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::step_size], time_label[::step_size]))]

        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=12)
        ax.set_xlabel('Frame / Time (h:m:s)', fontsize=20)
        
        #### 7. y축 세부설정
        ax.set_yticks(range(1, len(yticks)+1))
        ax.set_yticklabels(yticks, fontsize=30)	
        # ax.set_ylabel('Model', fontsize=36)

        #### 8. 범례 나타내기
        # legend_color = [Line2D([0], [0], color=self.RGB[i], lw=3, label=legend_name[i]) for i in range(2)]
        box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # ax.legend(handles=legend_color, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1, fontsize=20)
        
        return ax

    def _merge_data(self, gt, pred_list):
        data = np.zeros((len(gt), 1 + len(pred_list)))
        for di, d in enumerate([gt, *pred_list]):
            for di2, d2 in enumerate(d):
                data[di2, di] = d2
                
        return data

    def _find_pos_list(self, data):
        pos_info = []
        
        for i in range(data.shape[1]):
            _data = data[:, i]
            
            cum_val = _data[0]
            st_pos = 0
            ed_pos = 0
            X = []
            
            for d in _data[1:]:
                if d == cum_val:
                    ed_pos += 1
                else:
                    X.append([st_pos, ed_pos, cum_val])
                    cum_val = d
                    st_pos = ed_pos
                    ed_pos += 1
                    
            if ed_pos < len(data):
                X.append([st_pos, len(data), cum_val])
                
            pos_info.append(X)
            
        return pos_info
