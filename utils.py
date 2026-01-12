import numpy as np


def Z_ScoreNormalization(x):
    import numpy as np
    out = (x - np.mean(x)) / np.std(x)
    return out
def get_peaks_bestParameters(input, best_distance=1.0, best_prominence=0.53, best_rate=1.0):
    """
    Find peaks in the input signal with simplified parameters.

    Parameters:
    - input: array-like
        The input signal to process.
    - distance_factor: float, optional
        Factor to adjust the calculated distance between peaks. Default is 0.8.
    - initial_prominence: float, optional
        The prominence threshold for the first pass of peak filtering. Default is 0.35.
    - final_prominence: float, optional
        The prominence threshold for the final peak filtering. Default is 0.5.

    Returns:
    - pppeaks: array-like
        The indices of the final detected peaks.
    """
    from scipy import signal
    import numpy as np
    input = Z_ScoreNormalization(input)
    # 获取所有的峰值点
    peaks = signal.argrelextrema(input, np.greater)[0]
    peaks_h = input[peaks]
    peaks_h = Z_ScoreNormalization(peaks_h)
    # 根据峰值点的变化进行筛选
    pp = signal.find_peaks(peaks_h,distance=best_distance,height=0,prominence=best_prominence)
    pp = pp[0]
    ppeaks = peaks[pp]
    # print(int(len(ppeaks)), int(6000 / int(len(ppeaks))))
    # 根据大概的心跳点间隔进行寻找峰值点
    pppeaks = signal.find_peaks(input, height=0, distance=int(6000 / int(len(ppeaks))) * best_rate, prominence=0.5)[0]
    # pppeaks = delete_close(pppeaks, input)
    return pppeaks


# 检查路径是否存在，没有就创建
def make_dir_path(path):
    """Create directory if it doesn't exist and return the path"""
    import os

    if path is None:
        raise ValueError("Path cannot be None")
    
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return None
    

# 分割窗口
def split_data(data, b2b, window_size=6000, step=100):
    num_windows = (len(data) - window_size) // step
    hr= np.zeros((num_windows, 1))
    windows = np.zeros((num_windows, window_size))
    for i in range(num_windows):
        windows[i] = data[i * step:i * step + window_size]
        windows[i] = Z_ScoreNormalization(windows[i])
        hr[i] = b2b[i * step]
    return windows, hr



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch
from matplotlib.patches import Rectangle
def plt_plot(*args, **kwargs):
    def figure_para(ax, dic, i):
        
        ax.set_title(dic['title'],fontweight='bold') if 'title' in dic else None

        ax.text(0.01, 0.95, f'({chr(97+i)})', transform=ax.transAxes, va='top') if 'text' in dic else None

        ax.set_xlim(dic['xlim'][0], dic['xlim'][1]) if 'xlim' in dic else None

        ax.set_ylim(dic['ylim'][0], dic['ylim'][1]) if 'ylim' in dic else None

        ax.set_xlabel(dic['xlabel'],fontweight='bold') if 'xlabel' in dic else None

        ax.set_ylabel(dic['ylabel'],fontweight='bold') if 'ylabel' in dic else None
        
    def x_y(dic):

        # 保证y为list
        ys = [dic['y']] if not isinstance(dic['y'], list) else dic['y']
        # 保证x存在
        x = dic.get('x', np.array(range(len(ys[0]))))
        # 保证ls存在         
        ls = dic.get('l', dic.get('label', [None] * len(dic['y'])))
        ls = [ls] if not isinstance(ls, list) else ls
        # 保证linewidth存在
        lws = dic.get('linewidth', dic.get('lw', [2] * len(dic['y'])))
        lws = [lws] if not isinstance(lws, list) else lws
        # 保证cs存在    
        cs = dic.get('color', dic.get('c', [None] * len(dic['y'])))
        cs = [cs] if not isinstance(cs, list) else cs
        # 保证cs存在    
        cs = dic.get('color', dic.get('c', [None] * len(dic['y'])))
        cs = [cs] if not isinstance(cs, list) else cs
        # 保证alphas存在 
        alphas = dic.get('alpha', dic.get('a', [1] * len(dic['y'])))
        alphas = [alphas] if not isinstance(alphas, list) else alphas
        return [x, ys, ls, lws, cs, alphas]

    def figure_main(ax, dic, para_list, **kwargs):
        # 绘制多条曲线
        [x, ys, ls, lws, cs, alphas] = para_list
        [ax.plot(x, y, label=l, linewidth=lw, color=c, alpha=alpha) if l else ax.plot(x, y, label=l, linewidth=lw, color=c, alpha=alpha) for y, l, lw, c, alpha in zip(ys, ls, lws, cs, alphas)]
        ax.legend(fontsize=kwargs.get('legend_fontsize', 16)-4, loc=kwargs.get('legend_loc', 'best'))
        # ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=kwargs.get('legend_fontsize', 12)-4)
        # 绘制多条曲线点
        if 'points' in dic:
            ps = [dic['points']['p']] if not isinstance(dic['points']['p'], list) else dic['points']['p']
            pcs = dic['points'].get('color', dic['points'].get('c', ['red'] * len(ps)))
            pls = dic['points'].get('linewidth', dic['points'].get('lw', [1] * len(ps)))
            for y, p, pl, pc in zip(ys[:len(ps)], ps, pls, pcs):
                ax.scatter(x[p], y[p], linewidth=pl, color=pc)

            
        if 'hide' in dic:
            ax.set_xticks([]) if 'xticks' in dic['hide'] else None
            ax.set_xticklabels([]) if 'xticks' in dic['hide'] else None

            ax.set_yticks([]) if 'yticks' in dic['hide'] else None
            ax.set_yticklabels([]) if 'yticks' in dic['hide'] else None
            
            ax.spines['top'].set_color('none') if 'top' in dic['hide'] or 't' in dic['hide'] else None

            ax.spines['bottom'].set_color('none') if 'bottom' in dic['hide'] or 'b' in dic['hide'] else None

            ax.spines['left'].set_color('none') if 'left' in dic['hide'] or 'l' in dic['hide'] else None

            ax.spines['right'].set_color('none') if 'right' in dic['hide'] or 'r' in dic['hide'] else None
        
        # 打开网格线
        if 'grid' in dic:
            x_major_ticks_top = np.arange(np.min(x),np.max(x)+2,2)
            y_major_ticks_top = np.arange(np.min(ys),np.max(ys)+2,2)
            ax.set_xticks(x_major_ticks_top, minor=False)
            ax.set_yticks(y_major_ticks_top, minor=False)
            ax.grid(which="major", alpha=0.8,linestyle='dashed')
            
            x_minor_ticks_top = np.arange(np.min(x),np.max(x)+1,1)
            y_minor_ticks_top = np.arange(np.min(ys),np.max(ys)+2,1)
            ax.set_xticks(x_minor_ticks_top, minor=True)
            ax.set_yticks(y_minor_ticks_top, minor=True)
            ax.grid(which="minor", alpha=0.4, linestyle='dashed')
        
        # 绘制多条垂直虚线
        if 'vlines' in dic:
            vlines_linewidth = dic['vlines'].get('linewidth', dic['vlines'].get('lw', 1))
            vlines_color = dic['vlines'].get('color', dic['vlines'].get('c', 'red'))
            ymin, ymax = dic['vlines'].get('y', (ax.get_ylim()))
            ax.vlines(x[dic['vlines']['v']], ymin, ymax, colors=vlines_color, linestyles='dashed', linewidth=vlines_linewidth)
        # 创建一个方框，指定左上角的坐标、宽度和高度
        if 'rectangle' in dic:
            lw = dic['rectangle'].get('linewidth', dic['rectangle'].get('lw', 1))
            c = dic['rectangle'].get('color', dic['rectangle'].get('c', 'red'))
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            h = dic['rectangle'].get('h', ymax-ymin)
            w = dic['rectangle'].get('w', xmax-xmin)
            px = dic['rectangle'].get('px', xmin)
            py = dic['rectangle'].get('py', ymin)
            # h_lim = ymax-ymin + dic['rectangle'].get('h_lim') 
            # w_lim = xmax-xmin + dic['rectangle'].get('w_lim')
            # px_lim = xmin + dic['rectangle'].get('px_lim')
            # py_lim = ymin + dic['rectangle'].get('py_lim')
            ax.add_patch(Rectangle((px, py), w, h, fill=False, color=c, linestyle='dashed', linewidth=lw))

    def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                        x_ratio=0.05,y_ratio=0.05):
        """缩放内嵌图形，并且进行连线
        ax:         调用plt.subplots返回的画布。例如: fig,ax = plt.subplots(1,1)
        axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
        zone_left:  要放大区域的横坐标左端点
        zone_right: 要放大区域的横坐标右端点
        x:          X轴标签
        y:          列表,所有y值
        linked:     进行连线的位置，{'bottom','top','left','right'}
        x_ratio:    X轴缩放比例
        y_ratio:    Y轴缩放比例
        """
        xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
        xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

        y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
        ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
        ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

        axins.set_xlim(xlim_left, xlim_right)
        axins.set_ylim(ylim_bottom, ylim_top)

        ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
                [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

        if linked == 'bottom':
            xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
            xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
        elif  linked == 'top':
            xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
            xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
        elif  linked == 'left':
            xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
            xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
        elif  linked == 'right':
            xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
            xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
            
        con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                            coordsB="data",axesA=axins,axesB=ax)
        axins.add_artist(con)
        con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                            coordsB="data",axesA=axins,axesB=ax)
        axins.add_artist(con)

    def fig_save(fig, n, **kwargs):     
        if kwargs.get('savepath', False):
            fig.savefig(kwargs.get('savepath', False), dpi=kwargs.get('dpi', 500), bbox_inches='tight')
        if kwargs.get('save_path', False):
            fig.savefig(kwargs.get('save_path', False), dpi=kwargs.get('dpi', 500), bbox_inches='tight')
        # Show the plot
        if kwargs.get('show', True):
            plt.show()
            
    # res提取
    res = [arg for arg in args]
    n = len(res)
    
    # fig初始化
    plt.rcParams['font.size'] = kwargs.get('fontsize', 16)
    plt.rcParams['font.family'] = kwargs.get('fontfamily', 'Times New Roman')
    plt.rcParams['font.weight'] = kwargs.get('fontweight', 'bold')
    fig, axs = plt.subplots(n, 1, dpi=kwargs.get('dpi', 500), figsize=kwargs.get('figsize', (10, 2*n)), 
                            sharex=kwargs.get('sharex', False), squeeze=kwargs.get('squeeze', False))
    fig.subplots_adjust(hspace=kwargs.get('hspace', 0))
    fig.suptitle(kwargs.get('suptitle', None))
    
    # 多子图
    for i in range(n):
        dic = res[i]
        #define axs[i] axi
        if type(dic) == dict:
            # 图像参数初始化
            figure_para(axs[i, 0], dic, i)
            # 一图多线下确保形状一致
            para_list = x_y(dic)
            # 图像主图
            figure_main(axs[i, 0], dic, para_list, **kwargs)
            if 'in' in dic.keys(): 
                # 绘制缩放图
                axins = axs[i, 0].inset_axes(dic['in']['inset'])
                # # 缩放图参数初始化
                figure_para(axins, dic['in'], i)
                # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
                figure_main(axins, dic, para_list, **kwargs)
                # 局部显示并且进行连线
                x = para_list[0]
                ys = para_list[1]
                zone_and_linked(axs[i, 0], axins, dic['in']['section'][0], dic['in']['section'][1], x , ys, dic['in'].get('location', 'right'))
        else:
            axs[i, 0].plot(dic, linewidth=2)
    # fig存储和调整
    fig_save(fig,n, **kwargs)
    # Show the plot
    if kwargs.get('show', False):
        plt.show()


import numpy as np

def calculate_bcg_metrics(peak_indices, fs=100):
    # 将采样点索引转换为毫秒 (ms)
    # RR (或 JJ) 间隔
    peak_times_ms = np.array(peak_indices) / fs * 1000.0
    rr_intervals = np.diff(peak_times_ms)
    
    # 1. Mean RR (ms)
    mean_rr = np.mean(rr_intervals)
    
    # 2. SDNN (ms) - 使用样本标准差 (ddof=1)
    sdnn = np.std(rr_intervals, ddof=1)
    
    # 3. RMSSD (ms)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return {
        "Mean_RR_ms": round(mean_rr, 2),
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2),
        "BPM_from_MeanRR": round(60000 / mean_rr, 2) # 根据平均间期估算的 BPM
    }


def compt_B2B_rate_in_window(HB_locs_in_window, time_window_size):
    temp_beat_loc_1 = HB_locs_in_window[:-1]
    temp_beat_loc_2 = HB_locs_in_window[1:]
    temp_beat_loc_1 = np.array(temp_beat_loc_1)
    temp_beat_loc_2 = np.array(temp_beat_loc_2)
    # time_diff_inv = 100. / (temp_beat_loc_2 - temp_beat_loc_1)
    HB_rate_in_window = time_window_size / np.mean(temp_beat_loc_2 - temp_beat_loc_1)
    return HB_rate_in_window

def cmpt_HB_rate(HB_locs, time_sequence, time_window_size, method):
    HB_rate = np.zeros(len(time_sequence))

    temp_HB_locs_all = []
    for time_scan in range((time_sequence[0] + time_window_size), time_sequence[-1] + 1):
        temp_HB_locs1 = [
            i
            for i in HB_locs
            if ((i >= (time_scan - time_window_size + 1)) & (i <= time_scan))
        ]
        temp_HB_locs = np.unique(temp_HB_locs1)
        temp_HB_locs_all.append(temp_HB_locs)
        temp_numb_beats = len(temp_HB_locs) - 1

        if temp_numb_beats > 0:
            if method == "mean_beats":
                temp_window_length = (temp_HB_locs[-1] - temp_HB_locs[0] + 1) / 100
                HB_rate[time_scan] = temp_numb_beats * 60 / temp_window_length
            elif method == "B2B":
                # print("B2B",time_scan)
                HB_rate[time_scan-6000] = compt_B2B_rate_in_window(
                    temp_HB_locs, time_window_size
                )
    return HB_rate