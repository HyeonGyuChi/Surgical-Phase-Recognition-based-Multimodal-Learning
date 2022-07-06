# load exact visual kinematic featrue from recon kinematic set
from recon_kinematic_helper import get_bbox_loader, set_bbox_loader, get_bbox_obj_info, get_recon_method
import pandas as pd
import pickle

class recon_kinematic_filter():
    def __init__(self, task='PETRAW'):
        self.task = task

    def set_src_path(self, src_path):
        self.src_path = src_path

    def filtering(self, methods, extract_objs, extract_pairs):
        columns = [] # parse columns from recon_df

        obj_key, obj_to_color = get_bbox_obj_info(get_bbox_loader(self.task, target_path='', dsize='', sample_interval='')) # dummy (target_path, dsize, sample_interval)
        
        # print('\n[+] \tsetting filter columns ... \n\tmethod : {} \n\textract_objs : {} ==> {}\n'.format(methods, extract_objs, extract_pairs))
        # base
        entity_col = ['x_min', 'x_max', 'y_min', 'y_max'] # VOC style

        for obj in extract_objs: # (idx, bbox1 points + bbox2 points + ...)
            for i in range(len(obj_to_color[obj])):
                columns += ['{}_{}-{}'.format(obj, i, col) for col in entity_col]

        single_methods = [m for m in methods if m in ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity']]
        pair_methods = [m for m in methods if m in ['IoU', 'gIoU', 'dIoU', 'cIoU']]

        # single method
        for method in single_methods: # ['centroid', 'eoa', 'pathlen', ..]
            _, recon_method_col, _ = get_recon_method(method, img_size=(0,0)) # dummy (img_size)

            for obj in extract_objs: # ['Grasper', 'Blocks', 'obj3', ..]
                for i in range(len(obj_to_color[obj])):
                    target_entity = '{}_{}'.format(obj, i)
                    columns += ['{}-{}'.format(target_entity, col) for col in recon_method_col]

        # pair method
        for method in pair_methods: # ['IoU', 'gIoU', 'dIoU', 'cIoU']
            _, recon_method_col, _ = get_recon_method(method, img_size=(0,0)) # dummy (img_size)

            for src_obj, target_obj in extract_pairs: # ('Grasper', 'Grasper'), ('Grasper', 'Blocks') .. 
                for i in range(len(obj_to_color[src_obj])): # src per entiity
                    if src_obj == target_obj and i > 0 : break  # same obj, calc only one time

                    for j in range(len(obj_to_color[target_obj])): # target per entiity
                        if src_obj == target_obj and i == j: # don't calc with same entitiy
                            continue

                        src_entity = '{}_{}'.format(src_obj, i)
                        target_entity = '{}_{}'.format(target_obj, j)
                        columns += ['{}-{}-{}'.format(src_entity, target_entity, col) for col in recon_method_col]

        print('NUM OF FEATURE : {}'.format(len(columns)))
        
        print(columns)
        print('\n[-] \tsetting filter columns ...')

        print('\n[+] \tparsing from {} ...'.format(self.src_path))
        
        with open(self.src_path, 'rb') as f:
            src_data = pickle.load(f)

        print('\n ==> SOURCE')
        print('dshape:', src_data.shape)
        print('col num: ', len(src_data.columns))
        print('col', src_data.columns)

        filtered_data = src_data[columns] # parsing from colnb

        print('\n ==> FILTER')
        print('dshape:', filtered_data.shape)
        print(filtered_data)

        print('\n[-] \tparsing from {} ...\n'.format(self.src_path))

        return filtered_data
        

if __name__ == "__main__":
    methods = ['centroid', 'eoa', 'partial_pathlen', 'cumulate_pathlen', 'speed', 'velocity', 'IoU', 'gIoU', 'dIoU', 'cIoU']
    
    # extract_objs=['Grasper']
    # extract_pairs=[('Grasper', 'Grasper')]

    extract_objs=['Stapler_Head']
    # extract_objs=['HarmonicAce_Head', 'Endotip']
    extract_pairs=[('HarmonicAce_Head', 'MarylandBipolarForceps_Head'), ('HarmonicAce_Head', 'Endotip')]

    # src_path = '/dataset3/multimodal/PETRAW/Training/Seg_kine11/002_seg_ki.pkl'
    src_path = '/raid/multimodal/gastric/Kinematic_swin/R000001_seg_ki.pkl'
    rk_filter = recon_kinematic_filter(task='GASTRIC')
    rk_filter.set_src_path(src_path)
    filterd_data = rk_filter.filtering(methods, extract_objs, extract_pairs)

    filterd_data.to_csv('./test.csv')
        

'''
0: ('Background', 'HarmonicAce_Head')
1: ('Background', 'HarmonicAce_Body')
2: ('Background', 'MarylandBipolarForceps_Head')
3: ('Background', 'MarylandBipolarForceps_Wrist')
4: ('Background', 'MarylandBipolarForceps_Body')
5: ('Background', 'CadiereForceps_Head')
6: ('Background', 'CadiereForceps_Wrist')
7: ('Background', 'CadiereForceps_Body')
8: ('Background', 'CurvedAtraumaticGrasper_Head')
9: ('Background', 'CurvedAtraumaticGrasper_Body')
10: ('Background', 'Stapler_Head')
11: ('Background', 'Stapler_Body')
12: ('Background', 'Medium-LargeClipApplier_Head')
13: ('Background', 'Medium-LargeClipApplier_Wrist')
14: ('Background', 'Medium-LargeClipApplier_Body')
15: ('Background', 'SmallClipApplier_Head')
16: ('Background', 'SmallClipApplier_Wrist')
17: ('Background', 'SmallClipApplier_Body')
18: ('Background', 'Suction-Irrigation')
19: ('Background', 'Needle')
20: ('Background', 'Endotip')
21: ('Background', 'Specimenbag')
22: ('Background', 'DrainTube')
23: ('Background', 'Liver')
24: ('Background', 'Stomach')
25: ('Background', 'Pancreas')
26: ('Background', 'Spleen')
27: ('Background', 'Gallbbladder')
28: ('Background', 'Gauze')
29: ('Background', 'The_Other_Inst')
30: ('Background', 'The_Other_Tissue')
31: ('HarmonicAce_Head', 'HarmonicAce_Body')
32: ('HarmonicAce_Head', 'MarylandBipolarForceps_Head')
33: ('HarmonicAce_Head', 'MarylandBipolarForceps_Wrist')
34: ('HarmonicAce_Head', 'MarylandBipolarForceps_Body')
35: ('HarmonicAce_Head', 'CadiereForceps_Head')
36: ('HarmonicAce_Head', 'CadiereForceps_Wrist')
37: ('HarmonicAce_Head', 'CadiereForceps_Body')
38: ('HarmonicAce_Head', 'CurvedAtraumaticGrasper_Head')
39: ('HarmonicAce_Head', 'CurvedAtraumaticGrasper_Body')
40: ('HarmonicAce_Head', 'Stapler_Head')
41: ('HarmonicAce_Head', 'Stapler_Body')
42: ('HarmonicAce_Head', 'Medium-LargeClipApplier_Head')
43: ('HarmonicAce_Head', 'Medium-LargeClipApplier_Wrist')
44: ('HarmonicAce_Head', 'Medium-LargeClipApplier_Body')
45: ('HarmonicAce_Head', 'SmallClipApplier_Head')
46: ('HarmonicAce_Head', 'SmallClipApplier_Wrist')
47: ('HarmonicAce_Head', 'SmallClipApplier_Body')
48: ('HarmonicAce_Head', 'Suction-Irrigation')
49: ('HarmonicAce_Head', 'Needle')
50: ('HarmonicAce_Head', 'Endotip')
51: ('HarmonicAce_Head', 'Specimenbag')
52: ('HarmonicAce_Head', 'DrainTube')
53: ('HarmonicAce_Head', 'Liver')
54: ('HarmonicAce_Head', 'Stomach')
55: ('HarmonicAce_Head', 'Pancreas')
56: ('HarmonicAce_Head', 'Spleen')
57: ('HarmonicAce_Head', 'Gallbbladder')
58: ('HarmonicAce_Head', 'Gauze')
59: ('HarmonicAce_Head', 'The_Other_Inst')
60: ('HarmonicAce_Head', 'The_Other_Tissue')
61: ('HarmonicAce_Body', 'MarylandBipolarForceps_Head')
62: ('HarmonicAce_Body', 'MarylandBipolarForceps_Wrist')
63: ('HarmonicAce_Body', 'MarylandBipolarForceps_Body')
64: ('HarmonicAce_Body', 'CadiereForceps_Head')
65: ('HarmonicAce_Body', 'CadiereForceps_Wrist')
66: ('HarmonicAce_Body', 'CadiereForceps_Body')
67: ('HarmonicAce_Body', 'CurvedAtraumaticGrasper_Head')
68: ('HarmonicAce_Body', 'CurvedAtraumaticGrasper_Body')
69: ('HarmonicAce_Body', 'Stapler_Head')
70: ('HarmonicAce_Body', 'Stapler_Body')
71: ('HarmonicAce_Body', 'Medium-LargeClipApplier_Head')
72: ('HarmonicAce_Body', 'Medium-LargeClipApplier_Wrist')
73: ('HarmonicAce_Body', 'Medium-LargeClipApplier_Body')
74: ('HarmonicAce_Body', 'SmallClipApplier_Head')
75: ('HarmonicAce_Body', 'SmallClipApplier_Wrist')
76: ('HarmonicAce_Body', 'SmallClipApplier_Body')
77: ('HarmonicAce_Body', 'Suction-Irrigation')
78: ('HarmonicAce_Body', 'Needle')
79: ('HarmonicAce_Body', 'Endotip')
80: ('HarmonicAce_Body', 'Specimenbag')
81: ('HarmonicAce_Body', 'DrainTube')
82: ('HarmonicAce_Body', 'Liver')
83: ('HarmonicAce_Body', 'Stomach')
84: ('HarmonicAce_Body', 'Pancreas')
85: ('HarmonicAce_Body', 'Spleen')
86: ('HarmonicAce_Body', 'Gallbbladder')
87: ('HarmonicAce_Body', 'Gauze')
88: ('HarmonicAce_Body', 'The_Other_Inst')
89: ('HarmonicAce_Body', 'The_Other_Tissue')
90: ('MarylandBipolarForceps_Head', 'MarylandBipolarForceps_Wrist')
91: ('MarylandBipolarForceps_Head', 'MarylandBipolarForceps_Body')
92: ('MarylandBipolarForceps_Head', 'CadiereForceps_Head')
93: ('MarylandBipolarForceps_Head', 'CadiereForceps_Wrist')
94: ('MarylandBipolarForceps_Head', 'CadiereForceps_Body')
95: ('MarylandBipolarForceps_Head', 'CurvedAtraumaticGrasper_Head')
96: ('MarylandBipolarForceps_Head', 'CurvedAtraumaticGrasper_Body')
97: ('MarylandBipolarForceps_Head', 'Stapler_Head')
98: ('MarylandBipolarForceps_Head', 'Stapler_Body')
99: ('MarylandBipolarForceps_Head', 'Medium-LargeClipApplier_Head')
100: ('MarylandBipolarForceps_Head', 'Medium-LargeClipApplier_Wrist')
101: ('MarylandBipolarForceps_Head', 'Medium-LargeClipApplier_Body')
102: ('MarylandBipolarForceps_Head', 'SmallClipApplier_Head')
103: ('MarylandBipolarForceps_Head', 'SmallClipApplier_Wrist')
104: ('MarylandBipolarForceps_Head', 'SmallClipApplier_Body')
105: ('MarylandBipolarForceps_Head', 'Suction-Irrigation')
106: ('MarylandBipolarForceps_Head', 'Needle')
107: ('MarylandBipolarForceps_Head', 'Endotip')
108: ('MarylandBipolarForceps_Head', 'Specimenbag')
109: ('MarylandBipolarForceps_Head', 'DrainTube')
110: ('MarylandBipolarForceps_Head', 'Liver')
111: ('MarylandBipolarForceps_Head', 'Stomach')
112: ('MarylandBipolarForceps_Head', 'Pancreas')
113: ('MarylandBipolarForceps_Head', 'Spleen')
114: ('MarylandBipolarForceps_Head', 'Gallbbladder')
115: ('MarylandBipolarForceps_Head', 'Gauze')
116: ('MarylandBipolarForceps_Head', 'The_Other_Inst')
117: ('MarylandBipolarForceps_Head', 'The_Other_Tissue')
118: ('MarylandBipolarForceps_Wrist', 'MarylandBipolarForceps_Body')
119: ('MarylandBipolarForceps_Wrist', 'CadiereForceps_Head')
120: ('MarylandBipolarForceps_Wrist', 'CadiereForceps_Wrist')
121: ('MarylandBipolarForceps_Wrist', 'CadiereForceps_Body')
122: ('MarylandBipolarForceps_Wrist', 'CurvedAtraumaticGrasper_Head')
123: ('MarylandBipolarForceps_Wrist', 'CurvedAtraumaticGrasper_Body')
124: ('MarylandBipolarForceps_Wrist', 'Stapler_Head')
125: ('MarylandBipolarForceps_Wrist', 'Stapler_Body')
126: ('MarylandBipolarForceps_Wrist', 'Medium-LargeClipApplier_Head')
127: ('MarylandBipolarForceps_Wrist', 'Medium-LargeClipApplier_Wrist')
128: ('MarylandBipolarForceps_Wrist', 'Medium-LargeClipApplier_Body')
129: ('MarylandBipolarForceps_Wrist', 'SmallClipApplier_Head')
130: ('MarylandBipolarForceps_Wrist', 'SmallClipApplier_Wrist')
131: ('MarylandBipolarForceps_Wrist', 'SmallClipApplier_Body')
132: ('MarylandBipolarForceps_Wrist', 'Suction-Irrigation')
133: ('MarylandBipolarForceps_Wrist', 'Needle')
134: ('MarylandBipolarForceps_Wrist', 'Endotip')
135: ('MarylandBipolarForceps_Wrist', 'Specimenbag')
136: ('MarylandBipolarForceps_Wrist', 'DrainTube')
137: ('MarylandBipolarForceps_Wrist', 'Liver')
138: ('MarylandBipolarForceps_Wrist', 'Stomach')
139: ('MarylandBipolarForceps_Wrist', 'Pancreas')
140: ('MarylandBipolarForceps_Wrist', 'Spleen')
141: ('MarylandBipolarForceps_Wrist', 'Gallbbladder')
142: ('MarylandBipolarForceps_Wrist', 'Gauze')
143: ('MarylandBipolarForceps_Wrist', 'The_Other_Inst')
144: ('MarylandBipolarForceps_Wrist', 'The_Other_Tissue')
145: ('MarylandBipolarForceps_Body', 'CadiereForceps_Head')
146: ('MarylandBipolarForceps_Body', 'CadiereForceps_Wrist')
147: ('MarylandBipolarForceps_Body', 'CadiereForceps_Body')
148: ('MarylandBipolarForceps_Body', 'CurvedAtraumaticGrasper_Head')
149: ('MarylandBipolarForceps_Body', 'CurvedAtraumaticGrasper_Body')
150: ('MarylandBipolarForceps_Body', 'Stapler_Head')
151: ('MarylandBipolarForceps_Body', 'Stapler_Body')
152: ('MarylandBipolarForceps_Body', 'Medium-LargeClipApplier_Head')
153: ('MarylandBipolarForceps_Body', 'Medium-LargeClipApplier_Wrist')
154: ('MarylandBipolarForceps_Body', 'Medium-LargeClipApplier_Body')
155: ('MarylandBipolarForceps_Body', 'SmallClipApplier_Head')
156: ('MarylandBipolarForceps_Body', 'SmallClipApplier_Wrist')
157: ('MarylandBipolarForceps_Body', 'SmallClipApplier_Body')
158: ('MarylandBipolarForceps_Body', 'Suction-Irrigation')
159: ('MarylandBipolarForceps_Body', 'Needle')
160: ('MarylandBipolarForceps_Body', 'Endotip')
161: ('MarylandBipolarForceps_Body', 'Specimenbag')
162: ('MarylandBipolarForceps_Body', 'DrainTube')
163: ('MarylandBipolarForceps_Body', 'Liver')
164: ('MarylandBipolarForceps_Body', 'Stomach')
165: ('MarylandBipolarForceps_Body', 'Pancreas')
166: ('MarylandBipolarForceps_Body', 'Spleen')
167: ('MarylandBipolarForceps_Body', 'Gallbbladder')
168: ('MarylandBipolarForceps_Body', 'Gauze')
169: ('MarylandBipolarForceps_Body', 'The_Other_Inst')
170: ('MarylandBipolarForceps_Body', 'The_Other_Tissue')
171: ('CadiereForceps_Head', 'CadiereForceps_Wrist')
172: ('CadiereForceps_Head', 'CadiereForceps_Body')
173: ('CadiereForceps_Head', 'CurvedAtraumaticGrasper_Head')
174: ('CadiereForceps_Head', 'CurvedAtraumaticGrasper_Body')
175: ('CadiereForceps_Head', 'Stapler_Head')
176: ('CadiereForceps_Head', 'Stapler_Body')
177: ('CadiereForceps_Head', 'Medium-LargeClipApplier_Head')
178: ('CadiereForceps_Head', 'Medium-LargeClipApplier_Wrist')
179: ('CadiereForceps_Head', 'Medium-LargeClipApplier_Body')
180: ('CadiereForceps_Head', 'SmallClipApplier_Head')
181: ('CadiereForceps_Head', 'SmallClipApplier_Wrist')
182: ('CadiereForceps_Head', 'SmallClipApplier_Body')
183: ('CadiereForceps_Head', 'Suction-Irrigation')
184: ('CadiereForceps_Head', 'Needle')
185: ('CadiereForceps_Head', 'Endotip')
186: ('CadiereForceps_Head', 'Specimenbag')
187: ('CadiereForceps_Head', 'DrainTube')
188: ('CadiereForceps_Head', 'Liver')
189: ('CadiereForceps_Head', 'Stomach')
190: ('CadiereForceps_Head', 'Pancreas')
191: ('CadiereForceps_Head', 'Spleen')
192: ('CadiereForceps_Head', 'Gallbbladder')
193: ('CadiereForceps_Head', 'Gauze')
194: ('CadiereForceps_Head', 'The_Other_Inst')
195: ('CadiereForceps_Head', 'The_Other_Tissue')
196: ('CadiereForceps_Wrist', 'CadiereForceps_Body')
197: ('CadiereForceps_Wrist', 'CurvedAtraumaticGrasper_Head')
198: ('CadiereForceps_Wrist', 'CurvedAtraumaticGrasper_Body')
199: ('CadiereForceps_Wrist', 'Stapler_Head')
200: ('CadiereForceps_Wrist', 'Stapler_Body')
201: ('CadiereForceps_Wrist', 'Medium-LargeClipApplier_Head')
202: ('CadiereForceps_Wrist', 'Medium-LargeClipApplier_Wrist')
203: ('CadiereForceps_Wrist', 'Medium-LargeClipApplier_Body')
204: ('CadiereForceps_Wrist', 'SmallClipApplier_Head')
205: ('CadiereForceps_Wrist', 'SmallClipApplier_Wrist')
206: ('CadiereForceps_Wrist', 'SmallClipApplier_Body')
207: ('CadiereForceps_Wrist', 'Suction-Irrigation')
208: ('CadiereForceps_Wrist', 'Needle')
209: ('CadiereForceps_Wrist', 'Endotip')
210: ('CadiereForceps_Wrist', 'Specimenbag')
211: ('CadiereForceps_Wrist', 'DrainTube')
212: ('CadiereForceps_Wrist', 'Liver')
213: ('CadiereForceps_Wrist', 'Stomach')
214: ('CadiereForceps_Wrist', 'Pancreas')
215: ('CadiereForceps_Wrist', 'Spleen')
216: ('CadiereForceps_Wrist', 'Gallbbladder')
217: ('CadiereForceps_Wrist', 'Gauze')
218: ('CadiereForceps_Wrist', 'The_Other_Inst')
219: ('CadiereForceps_Wrist', 'The_Other_Tissue')
220: ('CadiereForceps_Body', 'CurvedAtraumaticGrasper_Head')
221: ('CadiereForceps_Body', 'CurvedAtraumaticGrasper_Body')
222: ('CadiereForceps_Body', 'Stapler_Head')
223: ('CadiereForceps_Body', 'Stapler_Body')
224: ('CadiereForceps_Body', 'Medium-LargeClipApplier_Head')
225: ('CadiereForceps_Body', 'Medium-LargeClipApplier_Wrist')
226: ('CadiereForceps_Body', 'Medium-LargeClipApplier_Body')
227: ('CadiereForceps_Body', 'SmallClipApplier_Head')
228: ('CadiereForceps_Body', 'SmallClipApplier_Wrist')
229: ('CadiereForceps_Body', 'SmallClipApplier_Body')
230: ('CadiereForceps_Body', 'Suction-Irrigation')
231: ('CadiereForceps_Body', 'Needle')
232: ('CadiereForceps_Body', 'Endotip')
233: ('CadiereForceps_Body', 'Specimenbag')
234: ('CadiereForceps_Body', 'DrainTube')
235: ('CadiereForceps_Body', 'Liver')
236: ('CadiereForceps_Body', 'Stomach')
237: ('CadiereForceps_Body', 'Pancreas')
238: ('CadiereForceps_Body', 'Spleen')
239: ('CadiereForceps_Body', 'Gallbbladder')
240: ('CadiereForceps_Body', 'Gauze')
241: ('CadiereForceps_Body', 'The_Other_Inst')
242: ('CadiereForceps_Body', 'The_Other_Tissue')
243: ('CurvedAtraumaticGrasper_Head', 'CurvedAtraumaticGrasper_Body')
244: ('CurvedAtraumaticGrasper_Head', 'Stapler_Head')
245: ('CurvedAtraumaticGrasper_Head', 'Stapler_Body')
246: ('CurvedAtraumaticGrasper_Head', 'Medium-LargeClipApplier_Head')
247: ('CurvedAtraumaticGrasper_Head', 'Medium-LargeClipApplier_Wrist')
248: ('CurvedAtraumaticGrasper_Head', 'Medium-LargeClipApplier_Body')
249: ('CurvedAtraumaticGrasper_Head', 'SmallClipApplier_Head')
250: ('CurvedAtraumaticGrasper_Head', 'SmallClipApplier_Wrist')
251: ('CurvedAtraumaticGrasper_Head', 'SmallClipApplier_Body')
252: ('CurvedAtraumaticGrasper_Head', 'Suction-Irrigation')
253: ('CurvedAtraumaticGrasper_Head', 'Needle')
254: ('CurvedAtraumaticGrasper_Head', 'Endotip')
255: ('CurvedAtraumaticGrasper_Head', 'Specimenbag')
256: ('CurvedAtraumaticGrasper_Head', 'DrainTube')
257: ('CurvedAtraumaticGrasper_Head', 'Liver')
258: ('CurvedAtraumaticGrasper_Head', 'Stomach')
259: ('CurvedAtraumaticGrasper_Head', 'Pancreas')
260: ('CurvedAtraumaticGrasper_Head', 'Spleen')
261: ('CurvedAtraumaticGrasper_Head', 'Gallbbladder')
262: ('CurvedAtraumaticGrasper_Head', 'Gauze')
263: ('CurvedAtraumaticGrasper_Head', 'The_Other_Inst')
264: ('CurvedAtraumaticGrasper_Head', 'The_Other_Tissue')
265: ('CurvedAtraumaticGrasper_Body', 'Stapler_Head')
266: ('CurvedAtraumaticGrasper_Body', 'Stapler_Body')
267: ('CurvedAtraumaticGrasper_Body', 'Medium-LargeClipApplier_Head')
268: ('CurvedAtraumaticGrasper_Body', 'Medium-LargeClipApplier_Wrist')
269: ('CurvedAtraumaticGrasper_Body', 'Medium-LargeClipApplier_Body')
270: ('CurvedAtraumaticGrasper_Body', 'SmallClipApplier_Head')
271: ('CurvedAtraumaticGrasper_Body', 'SmallClipApplier_Wrist')
272: ('CurvedAtraumaticGrasper_Body', 'SmallClipApplier_Body')
273: ('CurvedAtraumaticGrasper_Body', 'Suction-Irrigation')
274: ('CurvedAtraumaticGrasper_Body', 'Needle')
275: ('CurvedAtraumaticGrasper_Body', 'Endotip')
276: ('CurvedAtraumaticGrasper_Body', 'Specimenbag')
277: ('CurvedAtraumaticGrasper_Body', 'DrainTube')
278: ('CurvedAtraumaticGrasper_Body', 'Liver')
279: ('CurvedAtraumaticGrasper_Body', 'Stomach')
280: ('CurvedAtraumaticGrasper_Body', 'Pancreas')
281: ('CurvedAtraumaticGrasper_Body', 'Spleen')
282: ('CurvedAtraumaticGrasper_Body', 'Gallbbladder')
283: ('CurvedAtraumaticGrasper_Body', 'Gauze')
284: ('CurvedAtraumaticGrasper_Body', 'The_Other_Inst')
285: ('CurvedAtraumaticGrasper_Body', 'The_Other_Tissue')
286: ('Stapler_Head', 'Stapler_Body')
287: ('Stapler_Head', 'Medium-LargeClipApplier_Head')
288: ('Stapler_Head', 'Medium-LargeClipApplier_Wrist')
289: ('Stapler_Head', 'Medium-LargeClipApplier_Body')
290: ('Stapler_Head', 'SmallClipApplier_Head')
291: ('Stapler_Head', 'SmallClipApplier_Wrist')
292: ('Stapler_Head', 'SmallClipApplier_Body')
293: ('Stapler_Head', 'Suction-Irrigation')
294: ('Stapler_Head', 'Needle')
295: ('Stapler_Head', 'Endotip')
296: ('Stapler_Head', 'Specimenbag')
297: ('Stapler_Head', 'DrainTube')
298: ('Stapler_Head', 'Liver')
299: ('Stapler_Head', 'Stomach')
300: ('Stapler_Head', 'Pancreas')
301: ('Stapler_Head', 'Spleen')
302: ('Stapler_Head', 'Gallbbladder')
303: ('Stapler_Head', 'Gauze')
304: ('Stapler_Head', 'The_Other_Inst')
305: ('Stapler_Head', 'The_Other_Tissue')
306: ('Stapler_Body', 'Medium-LargeClipApplier_Head')
307: ('Stapler_Body', 'Medium-LargeClipApplier_Wrist')
308: ('Stapler_Body', 'Medium-LargeClipApplier_Body')
309: ('Stapler_Body', 'SmallClipApplier_Head')
310: ('Stapler_Body', 'SmallClipApplier_Wrist')
311: ('Stapler_Body', 'SmallClipApplier_Body')
312: ('Stapler_Body', 'Suction-Irrigation')
313: ('Stapler_Body', 'Needle')
314: ('Stapler_Body', 'Endotip')
315: ('Stapler_Body', 'Specimenbag')
316: ('Stapler_Body', 'DrainTube')
317: ('Stapler_Body', 'Liver')
318: ('Stapler_Body', 'Stomach')
319: ('Stapler_Body', 'Pancreas')
320: ('Stapler_Body', 'Spleen')
321: ('Stapler_Body', 'Gallbbladder')
322: ('Stapler_Body', 'Gauze')
323: ('Stapler_Body', 'The_Other_Inst')
324: ('Stapler_Body', 'The_Other_Tissue')
325: ('Medium-LargeClipApplier_Head', 'Medium-LargeClipApplier_Wrist')
326: ('Medium-LargeClipApplier_Head', 'Medium-LargeClipApplier_Body')
327: ('Medium-LargeClipApplier_Head', 'SmallClipApplier_Head')
328: ('Medium-LargeClipApplier_Head', 'SmallClipApplier_Wrist')
329: ('Medium-LargeClipApplier_Head', 'SmallClipApplier_Body')
330: ('Medium-LargeClipApplier_Head', 'Suction-Irrigation')
331: ('Medium-LargeClipApplier_Head', 'Needle')
332: ('Medium-LargeClipApplier_Head', 'Endotip')
333: ('Medium-LargeClipApplier_Head', 'Specimenbag')
334: ('Medium-LargeClipApplier_Head', 'DrainTube')
335: ('Medium-LargeClipApplier_Head', 'Liver')
336: ('Medium-LargeClipApplier_Head', 'Stomach')
337: ('Medium-LargeClipApplier_Head', 'Pancreas')
338: ('Medium-LargeClipApplier_Head', 'Spleen')
339: ('Medium-LargeClipApplier_Head', 'Gallbbladder')
340: ('Medium-LargeClipApplier_Head', 'Gauze')
341: ('Medium-LargeClipApplier_Head', 'The_Other_Inst')
342: ('Medium-LargeClipApplier_Head', 'The_Other_Tissue')
343: ('Medium-LargeClipApplier_Wrist', 'Medium-LargeClipApplier_Body')
344: ('Medium-LargeClipApplier_Wrist', 'SmallClipApplier_Head')
345: ('Medium-LargeClipApplier_Wrist', 'SmallClipApplier_Wrist')
346: ('Medium-LargeClipApplier_Wrist', 'SmallClipApplier_Body')
347: ('Medium-LargeClipApplier_Wrist', 'Suction-Irrigation')
348: ('Medium-LargeClipApplier_Wrist', 'Needle')
349: ('Medium-LargeClipApplier_Wrist', 'Endotip')
350: ('Medium-LargeClipApplier_Wrist', 'Specimenbag')
351: ('Medium-LargeClipApplier_Wrist', 'DrainTube')
352: ('Medium-LargeClipApplier_Wrist', 'Liver')
353: ('Medium-LargeClipApplier_Wrist', 'Stomach')
354: ('Medium-LargeClipApplier_Wrist', 'Pancreas')
355: ('Medium-LargeClipApplier_Wrist', 'Spleen')
356: ('Medium-LargeClipApplier_Wrist', 'Gallbbladder')
357: ('Medium-LargeClipApplier_Wrist', 'Gauze')
358: ('Medium-LargeClipApplier_Wrist', 'The_Other_Inst')
359: ('Medium-LargeClipApplier_Wrist', 'The_Other_Tissue')
360: ('Medium-LargeClipApplier_Body', 'SmallClipApplier_Head')
361: ('Medium-LargeClipApplier_Body', 'SmallClipApplier_Wrist')
362: ('Medium-LargeClipApplier_Body', 'SmallClipApplier_Body')
363: ('Medium-LargeClipApplier_Body', 'Suction-Irrigation')
364: ('Medium-LargeClipApplier_Body', 'Needle')
365: ('Medium-LargeClipApplier_Body', 'Endotip')
366: ('Medium-LargeClipApplier_Body', 'Specimenbag')
367: ('Medium-LargeClipApplier_Body', 'DrainTube')
368: ('Medium-LargeClipApplier_Body', 'Liver')
369: ('Medium-LargeClipApplier_Body', 'Stomach')
370: ('Medium-LargeClipApplier_Body', 'Pancreas')
371: ('Medium-LargeClipApplier_Body', 'Spleen')
372: ('Medium-LargeClipApplier_Body', 'Gallbbladder')
373: ('Medium-LargeClipApplier_Body', 'Gauze')
374: ('Medium-LargeClipApplier_Body', 'The_Other_Inst')
375: ('Medium-LargeClipApplier_Body', 'The_Other_Tissue')
376: ('SmallClipApplier_Head', 'SmallClipApplier_Wrist')
377: ('SmallClipApplier_Head', 'SmallClipApplier_Body')
378: ('SmallClipApplier_Head', 'Suction-Irrigation')
379: ('SmallClipApplier_Head', 'Needle')
380: ('SmallClipApplier_Head', 'Endotip')
381: ('SmallClipApplier_Head', 'Specimenbag')
382: ('SmallClipApplier_Head', 'DrainTube')
383: ('SmallClipApplier_Head', 'Liver')
384: ('SmallClipApplier_Head', 'Stomach')
385: ('SmallClipApplier_Head', 'Pancreas')
386: ('SmallClipApplier_Head', 'Spleen')
387: ('SmallClipApplier_Head', 'Gallbbladder')
388: ('SmallClipApplier_Head', 'Gauze')
389: ('SmallClipApplier_Head', 'The_Other_Inst')
390: ('SmallClipApplier_Head', 'The_Other_Tissue')
391: ('SmallClipApplier_Wrist', 'SmallClipApplier_Body')
392: ('SmallClipApplier_Wrist', 'Suction-Irrigation')
393: ('SmallClipApplier_Wrist', 'Needle')
394: ('SmallClipApplier_Wrist', 'Endotip')
395: ('SmallClipApplier_Wrist', 'Specimenbag')
396: ('SmallClipApplier_Wrist', 'DrainTube')
397: ('SmallClipApplier_Wrist', 'Liver')
398: ('SmallClipApplier_Wrist', 'Stomach')
399: ('SmallClipApplier_Wrist', 'Pancreas')
400: ('SmallClipApplier_Wrist', 'Spleen')
401: ('SmallClipApplier_Wrist', 'Gallbbladder')
402: ('SmallClipApplier_Wrist', 'Gauze')
403: ('SmallClipApplier_Wrist', 'The_Other_Inst')
404: ('SmallClipApplier_Wrist', 'The_Other_Tissue')
405: ('SmallClipApplier_Body', 'Suction-Irrigation')
406: ('SmallClipApplier_Body', 'Needle')
407: ('SmallClipApplier_Body', 'Endotip')
408: ('SmallClipApplier_Body', 'Specimenbag')
409: ('SmallClipApplier_Body', 'DrainTube')
410: ('SmallClipApplier_Body', 'Liver')
411: ('SmallClipApplier_Body', 'Stomach')
412: ('SmallClipApplier_Body', 'Pancreas')
413: ('SmallClipApplier_Body', 'Spleen')
414: ('SmallClipApplier_Body', 'Gallbbladder')
415: ('SmallClipApplier_Body', 'Gauze')
416: ('SmallClipApplier_Body', 'The_Other_Inst')
417: ('SmallClipApplier_Body', 'The_Other_Tissue')
418: ('Suction-Irrigation', 'Needle')
419: ('Suction-Irrigation', 'Endotip')
420: ('Suction-Irrigation', 'Specimenbag')
421: ('Suction-Irrigation', 'DrainTube')
422: ('Suction-Irrigation', 'Liver')
423: ('Suction-Irrigation', 'Stomach')
424: ('Suction-Irrigation', 'Pancreas')
425: ('Suction-Irrigation', 'Spleen')
426: ('Suction-Irrigation', 'Gallbbladder')
427: ('Suction-Irrigation', 'Gauze')
428: ('Suction-Irrigation', 'The_Other_Inst')
429: ('Suction-Irrigation', 'The_Other_Tissue')
430: ('Needle', 'Endotip')
431: ('Needle', 'Specimenbag')
432: ('Needle', 'DrainTube')
433: ('Needle', 'Liver')
434: ('Needle', 'Stomach')
435: ('Needle', 'Pancreas')
436: ('Needle', 'Spleen')
437: ('Needle', 'Gallbbladder')
438: ('Needle', 'Gauze')
439: ('Needle', 'The_Other_Inst')
440: ('Needle', 'The_Other_Tissue')
441: ('Endotip', 'Specimenbag')
442: ('Endotip', 'DrainTube')
443: ('Endotip', 'Liver')
444: ('Endotip', 'Stomach')
445: ('Endotip', 'Pancreas')
446: ('Endotip', 'Spleen')
447: ('Endotip', 'Gallbbladder')
448: ('Endotip', 'Gauze')
449: ('Endotip', 'The_Other_Inst')
450: ('Endotip', 'The_Other_Tissue')
451: ('Specimenbag', 'DrainTube')
452: ('Specimenbag', 'Liver')
453: ('Specimenbag', 'Stomach')
454: ('Specimenbag', 'Pancreas')
455: ('Specimenbag', 'Spleen')
456: ('Specimenbag', 'Gallbbladder')
457: ('Specimenbag', 'Gauze')
458: ('Specimenbag', 'The_Other_Inst')
459: ('Specimenbag', 'The_Other_Tissue')
460: ('DrainTube', 'Liver')
461: ('DrainTube', 'Stomach')
462: ('DrainTube', 'Pancreas')
463: ('DrainTube', 'Spleen')
464: ('DrainTube', 'Gallbbladder')
465: ('DrainTube', 'Gauze')
466: ('DrainTube', 'The_Other_Inst')
467: ('DrainTube', 'The_Other_Tissue')
468: ('Liver', 'Stomach')
469: ('Liver', 'Pancreas')
470: ('Liver', 'Spleen')
471: ('Liver', 'Gallbbladder')
472: ('Liver', 'Gauze')
473: ('Liver', 'The_Other_Inst')
474: ('Liver', 'The_Other_Tissue')
475: ('Stomach', 'Pancreas')
476: ('Stomach', 'Spleen')
477: ('Stomach', 'Gallbbladder')
478: ('Stomach', 'Gauze')
479: ('Stomach', 'The_Other_Inst')
480: ('Stomach', 'The_Other_Tissue')
481: ('Pancreas', 'Spleen')
482: ('Pancreas', 'Gallbbladder')
483: ('Pancreas', 'Gauze')
484: ('Pancreas', 'The_Other_Inst')
485: ('Pancreas', 'The_Other_Tissue')
486: ('Spleen', 'Gallbbladder')
487: ('Spleen', 'Gauze')
488: ('Spleen', 'The_Other_Inst')
489: ('Spleen', 'The_Other_Tissue')
490: ('Gallbbladder', 'Gauze')
491: ('Gallbbladder', 'The_Other_Inst')
492: ('Gallbbladder', 'The_Other_Tissue')
493: ('Gauze', 'The_Other_Inst')
494: ('Gauze', 'The_Other_Tissue')
495: ('The_Other_Inst', 'The_Other_Tissue')
'''